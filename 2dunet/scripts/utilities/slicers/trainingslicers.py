import glob
import logging
import os
from pathlib import Path

import numpy as np
from skimage import img_as_ubyte, io
from tqdm import tqdm
from utilities.base_data_manager import BaseDataManager
from utilities.base_data_utils import (
    axis_index_to_slice,
    get_axis_index_pairs,
    get_num_of_ims,
    get_numpy_from_path,
)


class TrainingDataSlicer(BaseDataManager):
    """Class that converts 3d data volumes into 2d image slices on disk for
    model training.
    Slicing is carried in all of the xy (z), xz (y) and yz (x) planes.

    Args:
        settings (SettingsData): An initialised SettingsData object.
    """

    def __init__(self, settings, data_vol_path, label_vol_path):
        super().__init__(data_vol_path, settings)
        self.data_im_out_dir = None
        self.seg_im_out_dir = None
        self.multilabel = False
        self.settings = settings
        self.label_vol_path = Path(label_vol_path)
        self.seg_vol = get_numpy_from_path(
            self.label_vol_path, internal_path=settings.seg_hdf5_path
        )
        self.preprocess_labels()

    def preprocess_labels(self):
        seg_classes = np.unique(self.seg_vol)
        self.num_seg_classes = len(seg_classes)
        if self.num_seg_classes > 2:
            self.multilabel = True
        logging.info(
            "Number of classes in segmentation dataset:" f" {self.num_seg_classes}"
        )
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
        for idx, current in enumerate(seg_classes):
            self.seg_vol[self.seg_vol == current] = idx

    def output_data_slices(self, data_dir, prefix):
        """Wrapper method to intitiate slicing data volume to disk.

        Args:
            data_dir (pathlib.Path): The path to the directory where images will be saved.
        """
        self.data_im_out_dir = data_dir
        logging.info("Slicing data volume and saving slices to disk")
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(self.data_vol, data_dir, prefix)

    def output_label_slices(self, data_dir, prefix):
        """Wrapper method to intitiate slicing label volume to disk.

        Args:
            data_dir (pathlib.Path): The path to the directory where images will be saved.
        """
        self.seg_im_out_dir = data_dir
        logging.info("Slicing label volume and saving slices to disk")
        os.makedirs(data_dir, exist_ok=True)
        self.output_slices_to_disk(self.seg_vol, data_dir, prefix, label=True)

    def output_slices_to_disk(self, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in the three orthogonal
        planes to images on disk.

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            label (bool): Whether this is a label volume.
        """
        shape_tup = data_arr.shape
        ax_idx_pairs = get_axis_index_pairs(shape_tup)
        num_ims = get_num_of_ims(shape_tup)
        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            out_path = output_path / f"{name_prefix}_{axis}_stack_{index}"
            self.output_im(axis_index_to_slice(data_arr, axis, index), out_path, label)

    def output_im(self, data, path, label=False):
        """Converts a slice of data into an image on disk.

        Args:
            data (numpy.array): The data slice to be converted.
            path (str): The path of the image file including the filename prefix.
            label (bool): Whether to convert values >1 to 1 for binary segmentation.
        """
        if label:
            if data.dtype != np.uint8:
                data = img_as_ubyte(data)
            if not self.multilabel:
                data[data > 1] = 1
        io.imsave(f"{path}.png", data)

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
        """Wrapper function that cleans up data and label image slices."""
        self.delete_data_im_slices()
        self.delete_label_im_slices()
