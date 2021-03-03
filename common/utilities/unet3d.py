# -*- coding: utf-8 -*-
"""Classes for 3d U-net training and prediction.
"""
import csv
import json
import logging
import os
import sys
import time
import math
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import termplotlib as tpl
import torchio
import torch.nn.functional as F
import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt
from skimage import exposure, img_as_ubyte, io
from tqdm import tqdm
import torch
from torch import nn as nn

from pytorch3dunet.unet3d.losses import BCEDiceLoss, DiceLoss, GeneralizedDiceLoss
from pytorch3dunet.unet3d.metrics import GenericAveragePrecision, MeanIoU
from pytorch3dunet.unet3d.model import ResidualUNet3D

from utilities.early_stopping import EarlyStopping

DEVICE_NUM = 0
BAR_FORMAT = "{l_bar}{bar: 30}{r_bar}{bar: -30b}"  # tqdm progress bar


class Unet3dTrainer:
    """Class that utlises 3d dataloaders to train a 3d Unet.

    Args:
        sampler
        settings
    """

    def __init__(self, sampler, settings):
        self.settings = settings
        # Params for learning rate finder
        self.lr_find_lr_diff = 7
        self.lr_find_loss_threshold = 0.05
        self.lr_find_adjust_value = 1
        self.starting_lr = float(settings.starting_lr)
        self.end_lr = float(settings.end_lr)
        self.log_lr_ratio = math.log(self.end_lr / self.starting_lr)
        self.lr_find_epochs = settings.lr_find_epochs
        # Params for model training
        self.num_epochs = settings.num_epochs
        self.patience = settings.patience
        self.loss_criterion = self.get_loss_criterion()
        self.eval_metric = self.get_eval_metric()
        self.model_struc_dict = settings.model
        self.model_struc_dict["out_channels"] = sampler.num_seg_classes
        # Get Data loaders
        self.training_loader = sampler.get_training_loader()
        self.validation_loader = sampler.get_validation_loader()
        # Set up model ready for training
        logging.info(f"Setting up the model on device {settings.cuda_device}.")
        self.model = self.create_unet_on_device(DEVICE_NUM, self.model_struc_dict)
        self.optimizer = self.create_optimizer(self.starting_lr)
        logging.info("Trainer created.")

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
            loss_criterion = DiceLoss(sigmoid_normalization=False)
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

    def create_unet_on_device(self, device_num, model_struc_dict):
        unet = ResidualUNet3D(**model_struc_dict)
        return unet.to(device_num)

    def create_optimizer(self, learning_rate):
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def prepare_batch(self, batch, device):
        inputs = batch["data"][torchio.DATA].to(device)
        targets = batch["label"][torchio.DATA]
        # One hot encode the channels
        targets = targets.squeeze()
        channels = []
        for label_num in range(self.model_struc_dict["out_channels"]):
            channel = torch.zeros_like(targets)
            channel[targets == label_num] = 1
            channels.append(channel)
        targets = torch.stack(channels, 1).to(device, dtype=torch.uint8)
        return inputs, targets

    def lr_finder(self, lr_scheduler, smoothing=0.05, plt_fig=True):
        lr_find_loss = []
        lr_find_lr = []
        iters = 0

        self.model.train()
        print(
            f"Training for {self.lr_find_epochs} epochs to create a learning "
            "rate plot."
        )
        for i in range(self.lr_find_epochs):
            for batch in tqdm(
                self.training_loader,
                desc=f"Epoch {i + 1}, batch number",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            ):
                inputs, targets = self.prepare_batch(batch, DEVICE_NUM)
                self.optimizer.zero_grad()
                output = self.model(inputs)
                if self.loss_criterion == "CrossEntropyLoss":
                    loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
                else:
                    loss = self.loss_criterion(output, targets)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
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

        if plt_fig:
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

    def find_appropriate_lr(self, lr_find_loss, lr_find_lr):
        """Function taken from
        https://forums.fast.ai/t/automated-learning-rate-suggester/44199
            Parameters:

            lr_diff provides the interval distance by units of the “index of LR”
            (log transform of LRs) between the right and left bound
            loss_threshold is the maximum difference between the left and right
            bound’s loss values to stop the shift
            adjust_value is a coefficient to the final learning rate for pure
            manual adjustment
        """
        # Get loss values and their corresponding gradients, and get lr values
        losses = np.array(lr_find_loss)
        assert self.lr_find_lr_diff < len(losses)
        loss_grad = np.gradient(losses)
        lrs = lr_find_lr

        # Search for index in gradients where loss is lowest before the loss spike
        # Initialize right and left idx using the lr_diff as a spacing unit
        # Set the local min lr as -1 to signify if threshold is too low
        local_min_lr = 0.00075  # Add as default value to fix bug
        r_idx = -1
        l_idx = r_idx - self.lr_find_lr_diff
        while (l_idx >= -len(losses)) and (
            abs(loss_grad[r_idx] - loss_grad[l_idx]) > self.lr_find_loss_threshold
        ):
            local_min_lr = lrs[l_idx]
            r_idx -= 1
            l_idx -= 1

        lr_to_use = local_min_lr * self.lr_find_adjust_value

        return lr_to_use

    def train_model(self, output_path, num_epochs, patience):
        """Performs training of model for a number of cycles
        with a learning rate that is determined automatically.
        """
        train_losses = []
        valid_losses = []
        eval_scores = []
        self.avg_train_losses = []  # per epoch training loss
        self.avg_valid_losses = []  #  per epoch validation loss
        self.avg_eval_scores = []  #  per epoch evaluation score

        def lr_exp_stepper(x):
            """Exponentially increase learning rate as part of strategy to find the
            optimum.
            Taken from
            https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
            """
            return math.exp(
                x
                * self.log_lr_ratio
                / (self.lr_find_epochs * len(self.training_loader))
            )

        logging.info("Finding learning rate for model.")
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_exp_stepper)
        lr_find_loss, lr_find_lr = self.lr_finder(lr_scheduler)
        lr_to_use = self.find_appropriate_lr(lr_find_loss, lr_find_lr)
        logging.info(f"LR to use {lr_to_use}")
        # Recreate model and start training
        logging.info("Recreating the U-net and optimizer.")
        self.model = self.create_unet_on_device(DEVICE_NUM, self.model_struc_dict)
        self.optimizer = self.create_optimizer(lr_to_use)
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=output_path,
            model_dict=self.model_struc_dict,
        )
        # Initialise the One Cycle learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr_to_use,
            steps_per_epoch=len(self.training_loader),
            epochs=num_epochs,
            pct_start=0.5,
        )

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            tic = time.perf_counter()
            logging.info(f"Epoch {epoch} of {num_epochs}")
            for batch in tqdm(
                self.training_loader, desc="Training batch", bar_format=BAR_FORMAT
            ):
                inputs, targets = self.prepare_batch(batch, DEVICE_NUM)
                self.optimizer.zero_grad()
                output = self.model(inputs)  # Forward pass
                if self.settings.loss_criterion == "CrossEntropyLoss":
                    loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
                else:
                    loss = self.loss_criterion(output, targets)
                loss.backward()  # Backward pass
                self.optimizer.step()
                lr_scheduler.step()  # update the learning rate
                train_losses.append(loss.item())  # record training loss

            self.model.eval()  # prep model for evaluation
            with torch.no_grad():
                for batch in tqdm(
                    self.validation_loader,
                    desc="Validation batch",
                    bar_format=BAR_FORMAT,
                ):
                    inputs, targets = self.prepare_batch(batch, DEVICE_NUM)
                    output = self.model(inputs)  # Forward pass
                    # calculate the loss
                    if self.settings.loss_criterion == "CrossEntropyLoss":
                        loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
                    else:
                        loss = self.loss_criterion(output, targets)
                    valid_losses.append(loss.item())  # record validation loss
                    s_max = nn.Softmax(dim=1)
                    probs = s_max(output)  # Convert the logits to probs
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

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(self.avg_valid_losses[-1], self.model)

            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

        # load the last checkpoint with the best model
        model_dict = torch.load(output_path, map_location="cpu")
        self.model.load_state_dict(model_dict["model_state_dict"])

    def output_loss_fig(self, model_out_path):
        
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.avg_train_losses)+1), self.avg_train_losses, label='Training Loss')
        plt.plot(range(1, len(self.avg_valid_losses)+1), self.avg_valid_losses, label='Validation Loss')

        minposs = self.avg_valid_losses.index(min(self.avg_valid_losses))+1 # find position of lowest validation loss
        plt.axvline(minposs, linestyle='--', color='r',
                    label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(self.avg_train_losses)+1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig_out_pth = f"{model_out_path.stem}_loss_plot.png"
        logging.info(f"Saving figure of training/validation losses to {fig_out_pth}")
        fig.savefig(fig_out_pth, bbox_inches='tight')
        # Output a list of training stats
        epoch_lst = range(len(self.avg_train_losses))
        rows = zip(epoch_lst, self.avg_train_losses, self.avg_valid_losses, self.avg_eval_scores)
        with open(f"{model_out_path.stem}_train_stats.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('Epoch', 'Train Loss', 'Valid Loss', 'Eval Score'))
            for row in rows:
                writer.writerow(row)


class Unet3dPredictor:

    def __init__(self, sampler, settings):
        self.settings = settings
        # Params for learning rate finder
        self.lr_find_lr_diff = 7
        self.lr_find_loss_threshold = 0.05
        self.lr_find_adjust_value = 1
        self.starting_lr = float(settings.starting_lr)
        self.end_lr = float(settings.end_lr)
        self.log_lr_ratio = math.log(self.end_lr / self.starting_lr)
        self.lr_find_epochs = settings.lr_find_epochs
        # Params for model training
        self.num_epochs = settings.num_epochs
        self.patience = settings.patience
        self.loss_criterion = self.get_loss_criterion()
        self.eval_metric = self.get_eval_metric()
        self.model_struc_dict = settings.model
        self.model_struc_dict["out_channels"] = sampler.num_seg_classes
        # Get Data loaders
        self.training_loader = sampler.get_training_loader()
        self.validation_loader = sampler.get_validation_loader()
        # Set up model ready for training
        logging.info(f"Setting up the model on device {settings.cuda_device}.")
        self.model = self.create_unet_on_device(DEVICE_NUM, self.model_struc_dict)
        self.optimizer = self.create_optimizer(self.starting_lr)
        logging.info("Trainer created.")

