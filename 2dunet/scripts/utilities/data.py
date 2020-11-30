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
from datetime import date
import re

import dask.array as da
import h5py as h5
import numpy as np
import yaml
from skimage import exposure, img_as_ubyte, img_as_float, io
from tqdm import tqdm
from fastai.vision import pil2tensor, crop_pad, Image

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

    def output_data_slices(self, data_dir):
        """Wrapper method to intitiate slicing data volume to disk.

        Args:
            data_dir (pathlib.Path): The path to the directory where images will be saved.
        """
        self.data_im_out_dir = data_dir
        logging.info(
            'Slicing data volume and saving slices to disk')
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(self.data_vol, data_dir, 'data')

    def output_label_slices(self, data_dir):
        """Wrapper method to intitiate slicing label volume to disk.

        Args:
            data_dir (pathlib.Path): The path to the directory where images will be saved.
        """
        self.seg_im_out_dir = data_dir
        logging.info(
            'Slicing label volume and saving slices to disk')
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(
            self.seg_vol, data_dir, 'seg', label=True)

    def output_slices_to_disk(self, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in the three orthogonal
        planes to images on disk. 
        
        Args:
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


class PredictionDataSlicer(DataSlicerBase):
    """Class that converts 3d data volumes into 2d image slices for
    segmentation prediction and that combines the slices back into volumes after
    prediction. 

    1. Slicing is carried in the xy (z), xz (y) and yz (x) planes. 2. The data
    volume is rotated by 90 degrees. Steps 1 and 2 are then repeated untill
    4 rotations have been sliced.

    The class also has methods to combine the image slices in to 3d volumes and
    also to combine these volumes and perform consensus thresholding.

    Args:
        settings (SettingsData): An initialised SettingsData object.
    """

    def __init__(self, settings, predictor):
        super().__init__(settings)
        self.consensus_vals = map(int, settings.consensus_vals)
        self.predictor = predictor

    def setup_folder_stucture(self, root_path):
        vol_dir= root_path/f'{date.today()}_predicted_volumes'
        non_rotated = vol_dir/f'{date.today()}_non_rotated_volumes'
        rot_90_seg = vol_dir/f'{date.today()}_rot_90_volumes'
        rot_180_seg = vol_dir/f'{date.today()}_rot_180_volumes'
        rot_270_seg = vol_dir/f'{date.today()}_rot_270_volumes'

        self.dir_list = [
            ('non_rotated', non_rotated),
            ('rot_90_seg', rot_90_seg),
            ('rot_180_seg', rot_180_seg),
            ('rot_270_seg', rot_270_seg)
        ]
        for _, dir_path in self.dir_list:
            os.makedirs(dir_path, exist_ok=True)

    def combine_slices_to_vol(self, folder_path):
        output_path_list = []
        file_list = folder_path.ls()
        axis_list = ['z', 'y', 'x']
        number_regex = re.compile(r'\_(\d+)\.png')
        for axis in axis_list:
            # Generate list of files for that axis
            axis_files = [x for x in file_list if re.search(
                f'\_({axis})\_', str(x))]
            logging.info(f'Axis {axis}: {len(axis_files)} files found, creating' \
                ' volume')
            # Load in the first image to get dimensions
            first_im = io.imread(axis_files[0])
            shape_tuple = first_im.shape
            z_dim = len(axis_files)
            y_dim, x_dim = shape_tuple
            data_vol = np.empty([z_dim, y_dim, x_dim], dtype=np.uint8)
            for filename in axis_files:
                m = number_regex.search(str(filename))
                index = int(m.group(1))
                im_data = io.imread(filename)
                data_vol[index, :, :] = im_data
            if axis == 'y':
                data_vol = np.swapaxes(data_vol, 0, 1)
            if axis == 'x':
                data_vol = np.swapaxes(data_vol, 0, 2)
                data_vol = np.swapaxes(data_vol, 0, 1)
            output_path = folder_path/f'{axis}_axis_seg_combined.h5'
            output_path_list.append(output_path)
            logging.info(f'Outputting {axis} axis volume to {output_path}')
            with h5.File(output_path, 'w') as f:
                f['/data'] = data_vol
            # Delete the images
            logging.info(f"Deleting {len(axis_files)} image files for axis {axis}")
            for filename in axis_files:
                os.remove(filename)
        return output_path_list

    def combine_vols(self, output_path_list, k, prefix, final=False):
        num_vols = len(output_path_list)
        combined = self.da_from_data(output_path_list[0])
        for subsequent in output_path_list[1:]:
            combined += self.da_from_data(subsequent)
        combined_out_path = output_path_list[0].parent.parent / \
            f'{date.today()}_{prefix}_{num_vols}_volumes_combined.h5'
        if final:
            combined_out_path = output_path_list[0].parent / \
                f'{date.today()}_{prefix}_12_volumes_combined.h5'
        logging.info(f'Outputting the {num_vols} combined volumes to {combined_out_path}')
        combined = combined.compute()
        combined = np.rot90(combined, 0 - k)
        with h5.File(combined_out_path, 'w') as f:
            f['/data'] = combined
        return combined_out_path

    def predict_single_slice(self, axis, index, data, output_path):
        data = img_as_float(data)
        img = Image(pil2tensor(data, dtype=np.float32))
        self.fix_odd_sides(img)
        prediction = self.predictor.model.predict(img)
        pred_slice = img_as_ubyte(prediction[1][0])
        io.imsave(
            output_path/f"unet_prediction_{axis}_stack_{index}.png", pred_slice)

    def fix_odd_sides(self, example_image):
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

    def predict_orthog_slices_to_disk(self, data_arr, output_path):
        """Outputs slices from data or ground truth seg volumes sliced in
         all three of the orthogonal planes"""
        shape_tup = data_arr.shape
        ax_idx_pairs = self.get_axis_index_pairs(shape_tup)
        num_ims = self.get_num_of_ims(shape_tup)
        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            self.predict_single_slice(
                axis, index, self.axis_index_to_slice(data_arr, axis, index), output_path)

    def consensus_threshold(self, input_path):
        for val in self.consensus_vals:
            combined = self.da_from_data(input_path)
            combined_out = input_path.parent / \
                f'{date.today()}_combined_consensus_thresh_cutoff_{val}.h5'
            combined[combined < val] = 0
            combined[combined >= val] = 255
            logging.info(f'Writing to {combined_out}')
            combined.to_hdf5(combined_out, '/data')

    def predict_12_ways(self, root_path):
        self.setup_folder_stucture(root_path)
        combined_vol_paths = []
        for k in tqdm(range(4), ncols=100, desc='Total progress', postfix="\n"):
            key, output_path = self.dir_list[k]
            logging.info(f'Rotating volume {k * 90} degrees')
            rotated = np.rot90(self.data_vol, k)
            logging.info("Predicting slices to disk.")
            self.predict_orthog_slices_to_disk(rotated, output_path)
            output_path_list = self.combine_slices_to_vol(output_path)
            fp = self.combine_vols(output_path_list, k, key)
            combined_vol_paths.append(fp)
        # Combine all the volumes
        final_combined = self.combine_vols(combined_vol_paths, 0, 'final', True)
        self.consensus_threshold(final_combined)
