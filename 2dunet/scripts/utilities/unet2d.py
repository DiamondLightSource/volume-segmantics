# -*- coding: utf-8 -*-
"""Classes for 2d U-net training and prediction.
"""
import json
import logging
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pytorch3dunet.unet3d.losses import GeneralizedDiceLoss
import torch
import torch.nn.functional as F
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.utils.mem import gpu_mem_get_free_no_cache
from fastai.vision import (SegmentationItemList, dice, get_transforms,
                           imagenet_stats, lr_find, models, open_image,
                           unet_learner, crop_pad)
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from skimage import exposure, img_as_ubyte, io
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class Unet2dTrainer:
    """Class that takes in 2d images and corresponding segmentations and
    trains a 2dUnet with a pretrained ResNet34 encoder.

    Args:
        data_im_out_dir (pathlib.Path): Path to directory containing image slices.
        seg_im_out_dir (pathlib.Path): Path to to directory containing label slices.
        codes (list of str): Names of the label classes, must be the same length as
        number of classes.
        im_size (int): Size of images to input to network.
        weight_decay (float): Value of the weight decay regularisation term to use.
    """

    def __init__(self, data_im_out_dir, seg_im_out_dir, codes, settings):
        self.data_dir = data_im_out_dir
        self.label_dir = seg_im_out_dir
        self.codes = codes
        self.multilabel = len(codes) > 2
        self.image_size = settings.image_size
        # Params for learning rate finder
        self.lr_find_lr_diff = 15
        self.lr_find_loss_threshold = 0.05
        self.lr_find_adjust_value = 1
        self.gdl = None
        if settings.use_gdl:
            self.gdl = GeneralizedDiceLoss(sigmoid_normalization=False)
        # Params for model training
        self.weight_decay = float(settings.weight_decay)
        self.num_cyc_frozen = settings.num_cyc_frozen
        self.num_cyc_unfrozen = settings.num_cyc_unfrozen
        self.pct_lr_inc = settings.pct_lr_inc
        # Set up model ready for training
        self.batch_size = self.get_batchsize()
        self.create_training_dataset()
        self.setup_metrics_and_loss()
        self.create_model()

    def setup_metrics_and_loss(self):
        """Sets instance attributes for loss function and evaluation metrics
        according to whether binary or multilabel segmentation is being
        performed. 
        """
        
        if self.multilabel:
            logging.info("Setting up for multilabel segmentation since there are "
                         f"{len(self.codes)} classes")
            self.metrics = self.accuracy
            self.monitor = 'accuracy'
            self.loss_func = None
        else:
            logging.info("Setting up for binary segmentation since there are "
                         f"{len(self.codes)} classes")
            self.metrics = [partial(dice, iou=True)]
            self.monitor = 'dice'
            self.loss_func = self.bce_loss
        # If Generalised dice loss is selected, overwrite loss function
        if self.gdl:
            logging.info("Using generalised dice loss.")
            self.loss_func = self.generalised_dice_loss
           
    def create_training_dataset(self):
        """Creates a fastai segmentation dataset and stores it as an instance
        attribute.
        """
        logging.info("Creating training dataset from saved images.")
        src = (SegmentationItemList.from_folder(self.data_dir)
               .split_by_rand_pct()
               .label_from_func(self.get_label_name, classes=self.codes))
        self.data = (src.transform(get_transforms(), size=self.image_size, tfm_y=True)
                     .databunch(bs=self.batch_size)
                     .normalize(imagenet_stats))

    def create_model(self):
        """Creates a deep learning model linked to the dataset and stores it as
        an instance attribute.
        """
        logging.info("Creating 2d U-net model for training.")
        self.model = unet_learner(self.data, models.resnet34, metrics=self.metrics,
                                  wd=self.weight_decay, loss_func=self.loss_func,
                                  callback_fns=[partial(CSVLogger,
                                                filename='unet_training_history',
                                                append=True),
                                                partial(SaveModelCallback,
                                                monitor=self.monitor, mode='max',
                                                name="best_unet_model")])

    def train_model(self):
        """Performs transfer learning training of model for a number of cycles
        with parameters frozen or unfrozen and a learning rate that is determined automatically.
        """
        if self.num_cyc_frozen > 0:
            logging.info("Finding learning rate for frozen Unet model.")
            lr_to_use = self.find_appropriate_lr()
            logging.info(
                f"Training frozen Unet for {self.num_cyc_frozen} cycles with learning rate of {lr_to_use}.")
            self.model.fit_one_cycle(self.num_cyc_frozen, slice(
                lr_to_use/50, lr_to_use), pct_start=self.pct_lr_inc)
        if self.num_cyc_unfrozen > 0:
            self.model.unfreeze()
            logging.info("Finding learning rate for unfrozen Unet model.")
            lr_to_use = self.find_appropriate_lr()
            logging.info(
                f"Training unfrozen Unet for {self.num_cyc_unfrozen} cycles with learning rate of {lr_to_use}.")
            self.model.fit_one_cycle(self.num_cyc_unfrozen, slice(
                lr_to_use/50, lr_to_use), pct_start=self.pct_lr_inc)

    def save_model_weights(self, model_filepath):
        """Saves the model weights to a specified location.

        Args:
            model_filepath (pathlib.Path): Full path to location to save model
            weights excluding file extension.
        """
        self.model.save(model_filepath)
        json_path = model_filepath.parent/f"{model_filepath.name}_codes.json"
        zip_path = model_filepath.with_suffix('.zip')
        logging.info(
            f"Zipping the model weights to: {zip_path}")
        with open(json_path, 'w') as jf:
            json.dump(self.codes, jf)
        with ZipFile(zip_path, mode='w') as zf:
            zf.write(json_path, arcname=json_path.name)
            zf.write(model_filepath.with_suffix('.pth'),
                     arcname=model_filepath.with_suffix('.pth').name)
        os.remove(json_path)
        os.remove(model_filepath.with_suffix('.pth'))

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
        # Remove the restriction on the model prediction size
        self.model.data.single_ds.tfmargs['size'] = None
        filename_list = self.data.valid_ds.items[:3]
        img_list = []
        pred_list = []
        gt_list = []
        for fn in filename_list:
            img_list.append(open_image(fn))
            gt_list.append(io.imread(self.get_label_name(fn)))
        for img in img_list:
            self.fix_odd_sides(img)
            pred_list.append(img_as_ubyte(self.model.predict(img)[1][0]))
        # Horrible conversion from Fastai image to unit8 data array
        img_list = [img_as_ubyte(exposure.rescale_intensity(
            x.data.numpy()[0, :, :])) for x in img_list]

        # Create the plot
        fig = plt.figure(figsize=(12, 12))
        columns = 3
        rows = 3
        j = 0
        for i in range(columns*rows)[::3]:
            img = img_list[j]
            gt = gt_list[j]
            pred = pred_list[j]
            col1 = fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img, cmap='gray')
            col2 = fig.add_subplot(rows, columns, i + 2)
            plt.imshow(gt, cmap='gray')
            col3 = fig.add_subplot(rows, columns, i + 3)
            plt.imshow(pred, cmap='gray')
            j += 1
            if i == 0:
                col1.title.set_text('Data')
                col2.title.set_text('Ground Truth')
                col3.title.set_text('Prediction')
        plt.suptitle(f"Predictions for {model_path.name}", fontsize=16)
        plt_out_pth = model_path.parent/f'{model_path.stem}_prediction_image.png'
        logging.info(f"Saving example image predictions to {plt_out_pth}")
        plt.savefig(plt_out_pth, dpi=300)

    def find_appropriate_lr(self):
        """Function taken from https://forums.fast.ai/t/automated-learning-rate-suggester/44199
        which attempts to automatically find a learning rate from the fastai lr_find function.
            
            Returns:
                float: A value for a sensible learning rate to use for training.

        """
        lr_find(self.model)
        #Get loss values and their corresponding gradients, and get lr values
        losses = np.array(self.model.recorder.losses)
        assert(self.lr_find_lr_diff < len(losses))
        loss_grad = np.gradient(losses)
        learning_rates = self.model.recorder.lrs

        #Search for index in gradients where loss is lowest before the loss spike
        #Initialize right and left idx using the lr_diff as a spacing unit
        #Set the local min lr as -1 to signify if threshold is too low
        local_min_lr = 0.001  # Add as default value to fix bug
        r_idx = -1
        l_idx = r_idx - self.lr_find_lr_diff
        while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx])
               > self.lr_find_loss_threshold):
            local_min_lr = learning_rates[l_idx]
            r_idx -= 1
            l_idx -= 1

        lr_to_use = local_min_lr * self.lr_find_adjust_value
        return lr_to_use

    def get_batchsize(self):
        """Provides an appropriate batch size based upon free GPU memory. 

        Returns:
            int: A batch size for model training.
        """
        gpu_free_mem = gpu_mem_get_free_no_cache()
        if gpu_free_mem > 8200:
            batch_size = 8
        else:
            batch_size = 4
        logging.info(f"Using batch size of {batch_size}, have {gpu_free_mem} MB" \
            " of GPU RAM free.")
        return batch_size

    def fix_odd_sides(self, example_image):
        """Replaces an an odd image dimension with an even dimension by padding.
    
        Taken from https://forums.fast.ai/t/segmentation-mask-prediction-on-different-input-image-sizes/44389/7.

        Args:
            example_image (fastai.vision.Image): The image to be fixed.
        """
        if (list(example_image.size)[0] % 2) != 0:
            example_image = crop_pad(example_image,
                                    size=(list(example_image.size)[
                                        0]+1, list(example_image.size)[1]),
                                    padding_mode='reflection')

        if (list(example_image.size)[1] % 2) != 0:
            example_image = crop_pad(example_image,
                                    size=(list(example_image.size)[0], list(
                                        example_image.size)[1] + 1),
                                    padding_mode='reflection')

    def bce_loss(self, logits, labels):
        """Function to calulate Binary Cross Entropy loss from predictions.

        Args:
            logits (torch.Tensor): output from network.
            labels (torch.Tensor): ground truth label values.

        Returns:
            torch.Tensor: The BCE loss calulated on the predictions.
        """
        logits = logits[:, 1, :, :].float()
        labels = labels.squeeze(1).float()
        return F.binary_cross_entropy_with_logits(logits, labels)

    def generalised_dice_loss(self, logits, labels):
        labels = F.one_hot(torch.squeeze(labels), len(self.codes))
        labels = labels.permute((0, 3, 1, 2))
        return self.gdl(logits, labels)
    
    def accuracy(self, input, target):
        """Calculates and accuracy metric between predictions and ground truth
        labels.

        Args:
            input (torch.Tensor): The predictions.
            target (torchTensor): The desired output (ground truth).

        Returns:
            [type]: [description]
        """
        target = target.squeeze(1)
        return (input.argmax(dim=1) == target).float().mean()
 
    def get_label_name(self, img_fname):
        """Converts a path fo an image slice to a path for corresponding label
        slice.

        Args:
            img_fname (pathlib.Path): Path to an image slice file.

        Returns:
            pathlib.Path: Path to the corresponding segmentation label slice file. 
        """
        return self.label_dir/f'{"seg" + img_fname.stem[4:]}{img_fname.suffix}'


class Unet2dPredictor:
    """Class that can either load in fastai 2d Unet model weights or take an
    instance of a trained fastai Unet learner. It can then predict 2d
    segmentations of image slices provided and save them to disk.
    """

    def __init__(self, root_dir, model_path=None):
        self.dummy_fns = ['data_z_stack_0.png', 'seg_z_stack_0.png']
        self.dummy_dir = root_dir/'dummy_imgs'
        self.root_dir = root_dir

    def create_dummy_files(self):
        logging.info(f"Creating dummy images in {self.dummy_dir}.")
        os.makedirs(self.dummy_dir, exist_ok=True)
        for fn in self.dummy_fns:
            dummy_im = np.random.randint(256, size=(256, 256))
            io.imsave(self.dummy_dir/fn, img_as_ubyte(dummy_im))

    def create_dummy_dataset(self):
        """Creates a fastai segmentation dataset and stores it as an instance
        attribute.
        """
        logging.info("Creating training dataset from dummy images.")
        src = (SegmentationItemList.from_folder(self.dummy_dir)
                .split_by_rand_pct()
               .label_from_func(self.get_label_name, classes=self.codes))
        self.data = (src.transform(get_transforms(), size=256, tfm_y=True)
                     .databunch()
                     .normalize(imagenet_stats))

    def get_label_name(self, img_fname):
        """Converts a path fo an image slice to a path for corresponding label
        slice.

        Args:
            img_fname (pathlib.Path): Path to an image slice file.

        Returns:
            pathlib.Path: Path to the corresponding segmentation label slice file. 
        """
        return self.dummy_dir/f'{"seg" + img_fname.stem[4:]}{img_fname.suffix}'

    def create_model_from_zip(self, weights_fn):
        """Creates a deep learning model linked to the dataset and stores it as
        an instance attribute.
        """
        weights_fn = weights_fn.resolve()
        logging.info(f"Unzipping the model weights and label classes from {weights_fn}")
        output_dir = "extracted_model_files"
        os.makedirs(output_dir, exist_ok=True)
        with ZipFile(weights_fn, mode='r') as zf:
            zf.extractall(output_dir)
        out_path = self.root_dir/output_dir
        # Load in the label classes from the json file
        with open(out_path/f"{weights_fn.stem}_codes.json") as jf:
            codes = json.load(jf)
        logging.info(f"{codes}")
        if isinstance(codes, dict):
            logging.info("Converting label dictionary into list.")
            self.codes = [f"label_val_{i}" for i in codes]
        else:
            self.codes = codes
        # Have to create dummy files and datset before loading in model weights 
        self.create_dummy_files()
        self.create_dummy_dataset()
        logging.info("Creating 2d U-net model for prediction.")
        self.model = unet_learner(
            self.data, models.resnet34, model_dir=out_path)
        logging.info("Loading in the saved weights.")
        self.model.load(weights_fn.stem)
        # Remove the restriction on the model prediction size
        self.model.data.single_ds.tfmargs['size'] = None

    def get_model_from_trainer(self, trainer):
        self.model = trainer.model
