#!/usr/bin/env python

import argparse
import logging
import os
import sys
import warnings
import yaml
from datetime import date

import numpy as np
import h5py as h5
from pathlib import Path
import dask.array as da
from skimage import exposure, img_as_ubyte, io
from tqdm import tqdm
from fastai.utils.mem import gpu_mem_get_free_no_cache
from fastai.vision import (SegmentationItemList, dice, get_transforms,
                          imagenet_stats, unet_learner, models, lr_find)
from fastai.callbacks import CSVLogger, SaveModelCallback
from functools import partial
import torch.nn.functional as F


warnings.filterwarnings("ignore", category=UserWarning)

def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/data/file.h5] [path/to/segmentation/file.h5]",
        description="Train a 2d U-net model on the 3d data and corresponding"
        "segmentation provided in the files."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('data_vol_path', metavar='Image data file path', type=str,
                        help='the path to an HDF5 file containing the imaging data volume.')
    parser.add_argument('seg_vol_path', metavar='Segmentation file path', type=str,
                        help='the path to an HDF5 file containing a segmented volume.')
    return parser


class DataSlicer:
    """Class that converts 3d data volumes into 2d image slices on disk.
    Slicing is carried in one or all of the xy (z), xz (y) and yz (x) planes.

    Args:
        data_vol_path (pathlib.Path): Path to the data volume HDF5 file.
        seg_vol_path (pathlib.Path): Path to the label volume HDF5 file.
        normalise (bool): If True clip and downsample image data to unit8.
        st_dev_factor (float): The number of std deviations to clip data to.
    """

    def __init__(self, data_vol_path, seg_vol_path, normalise, st_dev_factor):
        self.multilabel = False
        self.data_vol = self.da_from_data(data_vol_path)
        self.seg_vol = self.da_from_data(seg_vol_path)
        self.st_dev_factor = st_dev_factor
        if normalise:
            self.data_vol = self.clip_to_uint8(self.data_vol.compute())
        seg_classes = np.unique(self.seg_vol.compute())
        self.num_seg_classes = len(seg_classes)
        if self.num_seg_classes > 2:
            self.multilabel = True
        logging.info("Number of classes in segmentation dataset:"
         f" {self.num_seg_classes}")
        logging.info(f"These classes are: {seg_classes}")
        if seg_classes[0] != 0:
            logging.info("Fixing label classes.")
            self.fix_label_classes(seg_classes)
        self.codes = [f"label_val_{i}" for i in seg_classes]

    def da_from_data(self, path):
        """Returns a dask array when given a path to an HDF5 file.

        The data is assumed to be found in '/data' in the file.

        Args:
            path(pathlib.Path): The path to the HDF5 file.

        Returns:
            dask.array: A dask array object for the data stored in the HDF5 file.
        """
        f = h5.File(path, 'r')
        d = f['/data']
        return da.from_array(d, chunks='auto')

    def clip_to_uint8(self, data):
        """Clips data to a certain number of st_devs of the mean and reduces
        bit depth to uint8.

        Args:
            data(np.array): The data to be processed.

        Returns:
            np.array: A unit8 data array.
        """
        logging.info("Clipping data and converting to uint8.")
        data_st_dev = np.std(data)
        data_mean = np.mean(data)
        num_vox = np.prod(data.shape)
        lower_bound = data_mean - (data_st_dev * self.st_dev_factor)
        upper_bound = data_mean + (data_st_dev * self.st_dev_factor)
        gt_ub = (data > upper_bound).sum()
        lt_lb = (data < lower_bound).sum()
        logging.info(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
        logging.info(
            f"Number of voxels above upper bound to be clipped {gt_ub} - percentage {gt_ub/num_vox * 100:.3f}%")
        logging.info(
            f"Number of voxels below lower bound to be clipped {lt_lb} - percentage {lt_lb/num_vox * 100:.3f}%")
        data = np.clip(data, lower_bound, upper_bound)
        data = exposure.rescale_intensity(data, out_range='float')
        return img_as_ubyte(data)

    def fix_label_classes(self, seg_classes):
        """Changes the data values of classes in a segmented volume so that
        they start from zero.

        Args:
            seg_classes(list): An ascending list of the labels in the volume.
        """
        if isinstance(self.seg_vol, da.core.Array):
            self.seg_vol = self.seg_vol.compute()
        for idx, current in enumerate(seg_classes):
            self.seg_vol[self.seg_vol == current] = idx

    def output_data_slices(self, data_dir, axis):
        """Wrapper method to intitiate slicing data volume to disk.

        Args:
            data_dir (pathlib.Path): The path to the direcotry where images will be saved.
            axis (str): One of [all, x, y, or z]. Which planes to slice.
        """
        logging.info(
            'Slicing data volume and saving slices to disk')
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(axis, self.data_vol, data_dir, 'data')

    def output_label_slices(self, data_dir, axis):
        """Wrapper method to intitiate slicing label volume to disk.

        Args:
            data_dir (pathlib.Path): The path to the direcotry where images will be saved.
            axis (str): One of [all, x, y, or z]. Which planes to slice.
        """
        logging.info(
            'Slicing label volume and saving slices to disk')
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(axis, self.seg_vol, data_dir, 'seg', label=True)

    def output_slices_to_disk(self, axis, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in one or all of the three orthogonal
        planes to images on disk. 
        
        Args:
            axis (str): Which plane to slice the data in. Either 'x', 'y, 'z' or 'all'.
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            label (bool): Whether this is a label volume.
        """
        shape_tup = data_arr.shape
        # There has to be a cleverer way to do this!
        if axis in ['z', 'all']:
            logging.info('Outputting z stack')
            for val in tqdm(range(shape_tup[0])):
                out_path = output_path/f"{name_prefix}_z_stack_{val}"
                self.output_im(data_arr[val, :, :], out_path, label)
        if axis in ['y', 'all']:
            logging.info('Outputting y stack')
            for val in tqdm(range(shape_tup[1])):
                out_path = output_path/f"{name_prefix}_y_stack_{val}"
                self.output_im(data_arr[:, val, :], out_path, label)
        if axis in ['x', 'all']:
            logging.info('Outputting x stack')
            for val in tqdm(range(shape_tup[2])):
                out_path = output_path/f"{name_prefix}_x_stack_{val}"
                self.output_im(data_arr[:, :, val], out_path, label)
        if axis not in ['x', 'y', 'z', 'all']:
            logging.error("Axis should be one of: [all, x, y, or z]!")

    def output_im(self, data, path, label=False):
        """Converts a slice of data into an image on disk.
    
        Args:
            data (numpy.array): The data slice to be converted.
            path (str): The path of the image file including the filename prefix.
            label (bool): Whether to convert values >1 to 1 for binary segmentation.
        """
        if isinstance(data, da.core.Array):
            data = data.compute()
        if label and not self.multilabel:
            data[data > 1] = 1
        io.imsave(f'{path}.png', data)


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

    def __init__(self, data_im_out_dir, seg_im_out_dir, codes,
                im_size, weight_decay):
        self.data_dir = data_im_out_dir
        self.label_dir = seg_im_out_dir
        self.codes = codes
        self.multilabel = len(codes) > 2
        self.im_size = im_size
        self.weight_decay = weight_decay
        # Params for learning rate finder
        self.lr_find_lr_diff = 15
        self.lr_find_loss_threshold = 0.05
        self.lr_find_adjust_value = 1
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

    def create_training_dataset(self):
        """Creates a fastai segmentation dataset and stores it as an instance
        attribute.
        """
        logging.info("Creating training dataset from saved images.")
        src = (SegmentationItemList.from_folder(self.data_dir)
               .split_by_rand_pct()
               .label_from_func(self.get_label_name, classes=self.codes))
        self.data = (src.transform(get_transforms(), size=self.im_size, tfm_y=True)
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

    def train_model(self, num_cyc_frozen, num_cyc_unfrozen, pct_lr_inc):
        """Performs transfer learning training of model for a number of cycles
        with parameters frozen or unfrozen and a learning rate that is determined automatically.

        Args:
            num_cyc_frozen (int): Number of cycles to train just the network head.
            num_cyc_unfrozen (int): Number of cycles to train all trainable parameters.
            pct_lr_inc (float): Percentage of overall iterations where the LR is increasing.
        """
        if num_cyc_frozen > 0:
            logging.info("Finding learning rate for frozen Unet model.")
            lr_to_use = self.find_appropriate_lr()
            logging.info(
                f"Training frozen Unet for {num_cyc_frozen} cycles with learning rate of {lr_to_use}.")
            self.model.fit_one_cycle(num_cyc_frozen, slice(
                lr_to_use/50, lr_to_use), pct_start=pct_lr_inc)
        if num_cyc_unfrozen > 0:
            self.model.unfreeze()
            logging.info("Finding learning rate for unfrozen Unet model.")
            lr_to_use = self.find_appropriate_lr()
            logging.info(
                f"Training unfrozen Unet for {num_cyc_unfrozen} cycles with learning rate of {lr_to_use}.")
            self.model.fit_one_cycle(num_cyc_unfrozen, slice(
                lr_to_use/50, lr_to_use), pct_start=pct_lr_inc)

    def save_model_weights(self, model_filepath):
        """Saves the model weights to a specified location.

        Args:
            model_filepath (pathlib.Path): Full path to location to save model
            weights excluding file extension.
        """
        logging.info(f"Saving the model weights to: {model_filepath}")
        self.model.save(model_filepath)

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


class SettingsData:
    """Class to sanity check and then store settings from the commandline and
    a YAML settings file. Assumes given commandline args are filepaths.

    Args:
        settings_path (pathlib.Path): Path to the YAML file containing user settings.
        parser_args (argparse.Namespace): Parsed commandline arguments from argparse.
    """
    def __init__(self, settings_path, parser_args):
        logging.info(f"Loading settings from {settings_path}")
        if settings_path.exists():
            self.settings_path = settings_path
            with open(settings_path, 'r') as stream:
                self.settings_dict = yaml.safe_load(stream)
        else:
            logging.error("Couldn't find settings file... Exiting!")
            sys.exit(1)
        logging.debug(f"Commandline args given: {vars(parser_args)}")

        # Set the data as attributes, check paths are valid files
        for k, v in self.settings_dict.items():
            setattr(self, k, v)
        for k, v in vars(parser_args).items():
            # Check that files exist
            v = Path(v)
            if v.is_file():
                setattr(self, k, v)
            else:
                logging.error(f"The file {v} does not appear to exist. Exiting!")
                sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    root_path = Path.cwd()  # For module load script, use the CWD
    # Set up the settings
    parser = init_argparse()
    args = parser.parse_args()
    settings_path = Path(root_path, 'unet-settings', '2d_unet_train_settings.yaml')
    settings = SettingsData(settings_path, args)
    # Set up the DataSlicer and slice the data volumes into image files
    data_im_out_dir = root_path/settings.data_im_dirname # dir for data imgs
    seg_im_out_dir = root_path/settings.seg_im_out_dirname # dir for seg imgs
    slicer = DataSlicer(settings.data_vol_path, settings.seg_vol_path,
        settings.normalise, settings.st_dev_factor)
    slicer.output_data_slices(data_im_out_dir, axis='all')
    slicer.output_label_slices(seg_im_out_dir, axis='all')
    # Set up the UnetTrainer
    trainer = Unet2dTrainer(data_im_out_dir, seg_im_out_dir, slicer.codes,
                            settings.image_size, float(settings.weight_decay))
    # Train the model
    trainer.train_model(settings.num_cyc_frozen, settings.num_cyc_unfrozen,
                        settings.pct_lr_inc)
    # Save the model
    model_fn = f"{date.today()}_{settings.model_output_fn}"
    model_out = Path(root_path, model_fn)
    trainer.save_model_weights(model_out)
    # Save a figure showing the predicions
