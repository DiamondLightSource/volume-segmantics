# -*- coding: utf-8 -*-
"""Data utilities for U-net training and prediction.
"""
import glob
import logging
import os
import sys
import warnings
from itertools import chain, product
from pathlib import Path

import dask.array as da
import h5py as h5
import numpy as np
import yaml
from skimage import exposure, img_as_ubyte, io
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


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

class DataSlicerBase:
    """Base class for classes that convert 3d data volumes into 2d image slices on disk.
    Slicing is carried in all of the xy (z), xz (y) and yz (x) planes.

    Args:
        settings (SettingsData): An initialised SettingsData object.
    """

    def __init__(self, settings):
        self.st_dev_factor = settings.st_dev_factor
        self.data_vol = self.da_from_data(settings.data_vol_path)
        if settings.normalise:
            self.data_vol = self.clip_to_uint8(self.data_vol.compute())

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

    def get_axis_index_pairs(self, vol_shape):
        """Gets all combinations of axis and image slice index that are found
        in a 3d volume.

        Args:
            vol_shape (tuple): 3d volume shape (z, y, x)

        Returns:
            itertools.chain: An iterable containing all combinations of axis
            and image index that are found in the volume.
        """
        return chain(
            product('z', range(vol_shape[0])),
            product('y', range(vol_shape[1])),
            product('x', range(vol_shape[2]))
        )

    def axis_index_to_slice(self, vol, axis, index):
        """Converts an axis and image slice index for a 3d volume into a 2d 
        data array (slice). 

        Args:
            vol (3d array): The data volume to be sliced.
            axis (str): One of 'z', 'y' and 'x'.
            index (int): An image slice index found in that axis. 

        Returns:
            2d array: A 2d image slice corresponding to the axis and index.
        """
        if axis == 'z':
            return vol[index, :, :]
        if axis == 'y':
            return vol[:, index, :]
        if axis == 'x':
            return vol[:, :, index]

    def get_num_of_ims(self, vol_shape):
        """Calculates the total number of images that will be created when slicing
        an image volume in the z, y and x planes.

        Args:
            vol_shape (tuple): 3d volume shape (z, y, x).

        Returns:
            int: Total number of images that will be created when the volume is
            sliced. 
        """
        return sum(vol_shape)



class TrainingDataSlicer(DataSlicerBase):
    """Class that converts 3d data volumes into 2d image slices on disk for
    model training.
    Slicing is carried in all of the xy (z), xz (y) and yz (x) planes.

    Args:
        settings (SettingsData): An initialised SettingsData object.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.multilabel = False
        self.data_im_out_dir = None
        self.seg_im_out_dir = None
        self.seg_vol = self.da_from_data(settings.seg_vol_path)
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
        self.data_im_out_dir = data_dir
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
        self.seg_im_out_dir = data_dir
        logging.info(
            'Slicing label volume and saving slices to disk')
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(
            axis, self.seg_vol, data_dir, 'seg', label=True)

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
        ax_idx_pairs = self.get_axis_index_pairs(shape_tup)
        num_ims = self.get_num_of_ims(shape_tup)
        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            out_path = output_path/f"{name_prefix}_{axis}_stack_{index}"
            self.output_im(self.axis_index_to_slice(data_arr, axis, index),
                           out_path, label)

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

    def delete_data_im_slices(self):
        """Deletes image slices in the data image output directory. Leaves the
        directory in place since it contains model training history.
        """
        if self.data_im_out_dir:
            data_ims = glob.glob(f"{str(self.data_im_out_dir) + '/*.png'}")
            logging.info(f"Deleting {len(data_ims)} image slices")
            for fn in data_ims:
                os.remove(fn)

    def delete_label_im_slices(self):
        """Deletes label image slices in the segmented image output directory.
        Also deletes the directory itself.
        """
        if self.seg_im_out_dir:
            seg_ims = glob.glob(f"{str(self.seg_im_out_dir) + '/*.png'}")
            logging.info(f"Deleting {len(seg_ims)} segmentation slices")
            for fn in seg_ims:
                os.remove(fn)
            logging.info(f"Deleting the empty segmentation image directory")
            os.rmdir(self.seg_im_out_dir)

    def clean_up_slices(self):
        """Wrapper function that cleans up data and label image slices.
        """
        self.delete_data_im_slices()
        self.delete_label_im_slices()
