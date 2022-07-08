import csv
import logging
import math
import sys
import time
from typing import Union

import matplotlib as mpl
import numpy as np
import termplotlib as tpl

mpl.use("Agg")
import torch
import torch.nn as nn
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from matplotlib import pyplot as plt
from tqdm import tqdm
from volume_segmantics.data.dataloaders import get_2d_training_dataloaders
from volume_segmantics.data.pytorch3dunet_losses import (
    BCEDiceLoss,
    DiceLoss,
    GeneralizedDiceLoss,
)
from volume_segmantics.data.pytorch3dunet_metrics import (
    GenericAveragePrecision,
    MeanIoU,
)
from volume_segmantics.model.model_2d import create_model_on_device
from volume_segmantics.utilities.early_stopping import EarlyStopping


class VolSeg2dTrainer:
    """Class that utlises 2d dataloaders to train a 2d deep learning model.

    Args:
        sampler
        settings
    """

    def __init__(
        self, image_dir_path, label_dir_path, labels: Union[int, dict], settings
    ):
        self.training_loader, self.validation_loader = get_2d_training_dataloaders(
            image_dir_path, label_dir_path, settings
        )
        self.label_no = labels if isinstance(labels, int) else len(labels)
        self.codes = labels if isinstance(labels, dict) else {}
        self.settings = settings
        # Params for learning rate finder
        self.starting_lr = float(settings.starting_lr)
        self.end_lr = float(settings.end_lr)
        self.log_lr_ratio = self.calculate_log_lr_ratio()
        self.lr_find_epochs = settings.lr_find_epochs
        self.lr_reduce_factor = settings.lr_reduce_factor
        # Params for model training
        self.model_device_num = int(settings.cuda_device)
        self.patience = settings.patience
        self.loss_criterion = self.get_loss_criterion()
        self.eval_metric = self.get_eval_metric()
        self.model_struc_dict = self.get_model_struc_dict(settings)
        self.avg_train_losses = []  # per epoch training loss
        self.avg_valid_losses = []  #  per epoch validation loss
        self.avg_eval_scores = []  #  per epoch evaluation score

    def get_model_struc_dict(self, settings):
        model_struc_dict = settings.model
        model_type = utils.get_model_type(settings)
        model_struc_dict["type"] = model_type
        model_struc_dict["in_channels"] = cfg.MODEL_INPUT_CHANNELS
        model_struc_dict["classes"] = self.label_no
        return model_struc_dict

    def calculate_log_lr_ratio(self):
        return math.log(self.end_lr / self.starting_lr)

    def create_model_and_optimiser(self, learning_rate, frozen=False):
        logging.info(f"Setting up the model on device {self.settings.cuda_device}.")
        self.model = create_model_on_device(
            self.model_device_num, self.model_struc_dict
        )
        if frozen:
            self.freeze_model()
        logging.info(
            f"Model has {self.count_trainable_parameters()} trainable parameters, {self.count_parameters()} total parameters."
        )
        self.optimizer = self.create_optimizer(learning_rate)
        logging.info("Trainer created.")

    def freeze_model(self):
        logging.info(
            f"Freezing model with {self.count_trainable_parameters()} trainable parameters, {self.count_parameters()} total parameters."
        )
        for name, param in self.model.named_parameters():
            if all(["encoder" in name, "conv" in name]) and param.requires_grad:
                param.requires_grad = False

    def unfreeze_model(self):
        logging.info(
            f"Unfreezing model with {self.count_trainable_parameters()} trainable parameters, {self.count_parameters()} total parameters."
        )
        for name, param in self.model.named_parameters():
            if all(["encoder" in name, "conv" in name]) and not param.requires_grad:
                param.requires_grad = True

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_loss_criterion(self):
        if self.settings.loss_criterion == "BCEDiceLoss":
            alpha = self.settings.alpha
            beta = self.settings.beta
            logging.info(
                f"Using combined BCE and Dice loss with weighting of {alpha}*BCE "
                f"and {beta}*Dice"
            )
            loss_criterion = BCEDiceLoss(alpha, beta)
        elif self.settings.loss_criterion == "DiceLoss":
            logging.info("Using DiceLoss")
            loss_criterion = DiceLoss(normalization="none")
        elif self.settings.loss_criterion == "BCELoss":
            logging.info("Using BCELoss")
            loss_criterion = nn.BCEWithLogitsLoss()
        elif self.settings.loss_criterion == "CrossEntropyLoss":
            logging.info("Using CrossEntropyLoss")
            loss_criterion = nn.CrossEntropyLoss()
        elif self.settings.loss_criterion == "GeneralizedDiceLoss":
            logging.info("Using GeneralizedDiceLoss")
            loss_criterion = GeneralizedDiceLoss()
        else:
            logging.error("No loss criterion specified, exiting")
            sys.exit(1)
        return loss_criterion

    def get_eval_metric(self):
        # Get evaluation metric
        if self.settings.eval_metric == "MeanIoU":
            logging.info("Using MeanIoU")
            eval_metric = MeanIoU()
        elif self.settings.eval_metric == "GenericAveragePrecision":
            logging.info("Using GenericAveragePrecision")
            eval_metric = GenericAveragePrecision()
        else:
            logging.error("No evaluation metric specified, exiting")
            sys.exit(1)
        return eval_metric

    def train_model(self, output_path, num_epochs, patience, create=True, frozen=False):
        """Performs training of model for a number of cycles
        with a learning rate that is determined automatically.
        """
        train_losses = []
        valid_losses = []
        eval_scores = []

        if create:
            self.create_model_and_optimiser(self.starting_lr, frozen=frozen)
            lr_to_use = self.run_lr_finder()
            # Recreate model and start training
            self.create_model_and_optimiser(lr_to_use, frozen=frozen)
            early_stopping = self.create_early_stopping(output_path, patience)
        else:
            # Reduce starting LR, since model alreadiy partiallly trained
            self.starting_lr /= self.lr_reduce_factor
            self.end_lr /= self.lr_reduce_factor
            self.log_lr_ratio = self.calculate_log_lr_ratio()
            self.load_in_model_and_optimizer(
                self.starting_lr, output_path, frozen=frozen, optimizer=False
            )
            lr_to_use = self.run_lr_finder()
            min_loss = self.load_in_model_and_optimizer(
                self.starting_lr, output_path, frozen=frozen, optimizer=False
            )
            early_stopping = self.create_early_stopping(
                output_path, patience, best_score=-min_loss
            )

        # Initialise the One Cycle learning rate scheduler
        lr_scheduler = self.create_oc_lr_scheduler(num_epochs, lr_to_use)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            tic = time.perf_counter()
            logging.info(f"Epoch {epoch} of {num_epochs}")
            for batch in tqdm(
                self.training_loader,
                desc="Training batch",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                loss = self.train_one_batch(lr_scheduler, batch)
                train_losses.append(loss.item())  # record training loss

            self.model.eval()  # prep model for evaluation
            with torch.no_grad():
                for batch in tqdm(
                    self.validation_loader,
                    desc="Validation batch",
                    bar_format=cfg.TQDM_BAR_FORMAT,
                ):
                    inputs, targets = utils.prepare_training_batch(
                        batch, self.model_device_num, self.label_no
                    )
                    output = self.model(inputs)  # Forward pass
                    # calculate the loss
                    if self.settings.loss_criterion == "CrossEntropyLoss":
                        loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
                    else:
                        loss = self.loss_criterion(output, targets.float())
                    valid_losses.append(loss.item())  # record validation loss
                    s_max = nn.Softmax(dim=1)
                    probs = s_max(output)  # Convert the logits to probs
                    probs = torch.unsqueeze(probs, 2)
                    targets = torch.unsqueeze(targets, 2)
                    eval_score = self.eval_metric(probs, targets)
                    eval_scores.append(eval_score)  # record eval metric

            toc = time.perf_counter()
            # calculate average loss/metric over an epoch
            self.avg_train_losses.append(np.average(train_losses))
            self.avg_valid_losses.append(np.average(valid_losses))
            self.avg_eval_scores.append(np.average(eval_scores))
            logging.info(
                f"Epoch {epoch}. Training loss: {self.avg_train_losses[-1]}, Validation Loss: "
                f"{self.avg_valid_losses[-1]}. {self.settings.eval_metric}: {self.avg_eval_scores[-1]}"
            )
            logging.info(f"Time taken for epoch {epoch}: {toc - tic:0.2f} seconds")
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            eval_scores = []

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(
                self.avg_valid_losses[-1], self.model, self.optimizer, self.codes
            )

            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

        # load the last checkpoint with the best model
        self.load_in_weights(output_path)

    def load_in_model_and_optimizer(
        self, learning_rate, output_path, frozen=False, optimizer=False
    ):
        self.create_model_and_optimiser(learning_rate, frozen=frozen)
        logging.info("Loading in weights from saved checkpoint.")
        loss_val = self.load_in_weights(output_path, optimizer=optimizer)
        return loss_val

    def load_in_weights(self, output_path, optimizer=False, gpu=True):
        # load the last checkpoint with the best model
        if gpu:
            map_location = f"cuda:{self.model_device_num}"
        else:
            map_location = "cpu"
        model_dict = torch.load(output_path, map_location=map_location)
        logging.info("Loading model weights.")
        self.model.load_state_dict(model_dict["model_state_dict"])
        if optimizer:
            logging.info("Loading optimizer weights.")
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        return model_dict.get("loss_val", np.inf)

    def run_lr_finder(self):
        logging.info("Finding learning rate for model.")
        lr_scheduler = self.create_exponential_lr_scheduler()
        lr_find_loss, lr_find_lr = self.lr_finder(lr_scheduler)
        lr_to_use = self.find_lr_from_graph(lr_find_loss, lr_find_lr)
        logging.info(f"LR to use {lr_to_use}")
        return lr_to_use

    def lr_finder(self, lr_scheduler, smoothing=0.05):
        lr_find_loss = []
        lr_find_lr = []
        iters = 0

        self.model.train()
        logging.info(
            f"Training for {self.lr_find_epochs} epochs to create a learning "
            "rate plot."
        )
        for i in range(self.lr_find_epochs):
            for batch in tqdm(
                self.training_loader,
                desc=f"Epoch {i + 1}, batch number",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                loss = self.train_one_batch(lr_scheduler, batch)
                lr_step = self.optimizer.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                if iters == 0:
                    lr_find_loss.append(loss)
                else:
                    loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                    lr_find_loss.append(loss)
                if loss > 1 and iters > len(self.training_loader) // 1.333:
                    break
                iters += 1

        if self.settings.plot_lr_graph:
            fig = tpl.figure()
            fig.plot(
                np.log10(lr_find_lr),
                lr_find_loss,
                width=50,
                height=30,
                xlabel="Log10 Learning Rate",
            )
            fig.show()

        return lr_find_loss, lr_find_lr

    @staticmethod
    def find_lr_from_graph(
        lr_find_loss: torch.Tensor, lr_find_lr: torch.Tensor
    ) -> float:
        """Calculates learning rate corresponsing to minimum gradient in graph
        of loss vs learning rate.

        Args:
            lr_find_loss (torch.Tensor): Loss values accumulated during training
            lr_find_lr (torch.Tensor): Learning rate used for mini-batch

        Returns:
            float: The learning rate at the point when loss was falling most steeply
            divided by a fudge factor.
        """
        default_min_lr = cfg.DEFAULT_MIN_LR  # Add as default value to fix bug
        # Get loss values and their corresponding gradients, and get lr values
        for i in range(0, len(lr_find_loss)):
            if lr_find_loss[i].is_cuda:
                lr_find_loss[i] = lr_find_loss[i].cpu()
            lr_find_loss[i] = lr_find_loss[i].detach().numpy()
        losses = np.array(lr_find_loss)
        try:
            gradients = np.gradient(losses)
            min_gradient = gradients.min()
            if min_gradient < 0:
                min_loss_grad_idx = gradients.argmin()
            else:
                logging.info(
                    f"Minimum gradient: {min_gradient} was positive, returning default value instead."
                )
                return default_min_lr
        except Exception as e:
            logging.info(f"Failed to compute gradients, returning default value. {e}")
            return default_min_lr
        min_lr = lr_find_lr[min_loss_grad_idx]
        return min_lr / cfg.LR_DIVISOR

    def lr_exp_stepper(self, x):
        """Exponentially increase learning rate as part of strategy to find the
        optimum.
        Taken from
        https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
        """
        return math.exp(
            x * self.log_lr_ratio / (self.lr_find_epochs * len(self.training_loader))
        )

    def create_optimizer(self, learning_rate):
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def create_exponential_lr_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_exp_stepper)

    def create_oc_lr_scheduler(self, num_epochs, lr_to_use):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr_to_use,
            steps_per_epoch=len(self.training_loader),
            epochs=num_epochs,
            pct_start=self.settings.pct_lr_inc,
        )

    def create_early_stopping(self, output_path, patience, best_score=None):
        return EarlyStopping(
            patience=patience,
            verbose=True,
            path=output_path,
            model_dict=self.model_struc_dict,
            best_score=best_score,
        )

    def train_one_batch(self, lr_scheduler, batch):
        inputs, targets = utils.prepare_training_batch(
            batch, self.model_device_num, self.label_no
        )
        self.optimizer.zero_grad()
        output = self.model(inputs)  # Forward pass
        if self.settings.loss_criterion == "CrossEntropyLoss":
            loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
        else:
            loss = self.loss_criterion(output, targets.float())
        loss.backward()  # Backward pass
        self.optimizer.step()
        lr_scheduler.step()  # update the learning rate
        return loss

    def output_loss_fig(self, model_out_path):

        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            range(1, len(self.avg_train_losses) + 1),
            self.avg_train_losses,
            label="Training Loss",
        )
        plt.plot(
            range(1, len(self.avg_valid_losses) + 1),
            self.avg_valid_losses,
            label="Validation Loss",
        )

        minposs = (
            self.avg_valid_losses.index(min(self.avg_valid_losses)) + 1
        )  # find position of lowest validation loss
        plt.axvline(
            minposs, linestyle="--", color="r", label="Early Stopping Checkpoint"
        )

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.xlim(0, len(self.avg_train_losses) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_dir = model_out_path.parent
        fig_out_pth = output_dir / f"{model_out_path.stem}_loss_plot.png"
        logging.info(f"Saving figure of training/validation losses to {fig_out_pth}")
        fig.savefig(fig_out_pth, bbox_inches="tight")
        # Output a list of training stats
        epoch_lst = range(len(self.avg_train_losses))
        rows = zip(
            epoch_lst,
            self.avg_train_losses,
            self.avg_valid_losses,
            self.avg_eval_scores,
        )
        with open(output_dir / f"{model_out_path.stem}_train_stats.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("Epoch", "Train Loss", "Valid Loss", "Eval Score"))
            for row in rows:
                writer.writerow(row)

    def output_prediction_figure(self, model_path):
        """Saves a figure containing image slice data for three random images
        fromthe validation dataset along with the corresponding ground truth
        label image and corresponding prediction output from the model attached
        to this class instance. The image is saved to the same directory as the
        model weights.

        Args:
            model_path (pathlib.Path): Full path to the model weights file,
            this is used to get the directory and name of the model not to
            load and predict.
        """
        self.model.eval()  # prep model for evaluation
        batch = next(iter(self.validation_loader))  # Get first batch
        with torch.no_grad():
            inputs, targets = utils.prepare_training_batch(
                batch, self.model_device_num, self.label_no
            )
            output = self.model(inputs)  # Forward pass
            s_max = nn.Softmax(dim=1)
            probs = s_max(output)  # Convert the logits to probs
            labels = torch.argmax(probs, dim=1)  # flatten channels

        # Create the plot
        bs = self.validation_loader.batch_size
        if bs < 4:
            rows = bs
        else:
            rows = 4
        fig = plt.figure(figsize=(12, 16))
        columns = 3
        j = 0
        for i in range(columns * rows)[::3]:
            img = inputs[j].squeeze().cpu()
            gt = torch.argmax(targets[j], dim=0).cpu()
            pred = labels[j].cpu()
            col1 = fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img, cmap="gray")
            col2 = fig.add_subplot(rows, columns, i + 2)
            plt.imshow(gt, cmap="gray")
            col3 = fig.add_subplot(rows, columns, i + 3)
            plt.imshow(pred, cmap="gray")
            j += 1
            if i == 0:
                col1.title.set_text("Data")
                col2.title.set_text("Ground Truth")
                col3.title.set_text("Prediction")
        plt.suptitle(f"Predictions for {model_path.name}", fontsize=16)
        plt_out_pth = model_path.parent / f"{model_path.stem}_prediction_image.png"
        logging.info(f"Saving example image predictions to {plt_out_pth}")
        plt.savefig(plt_out_pth, dpi=300)
