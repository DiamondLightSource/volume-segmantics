import logging
import sys
from pathlib import Path

import h5py as h5
import numpy as np
import yaml
from skimage import exposure, img_as_float, img_as_ubyte, io
from skimage.measure import block_reduce


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
            with open(settings_path, "r") as stream:
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


class BaseDataUtils:
    def __init__(self, settings):
        self.st_dev_factor = settings.st_dev_factor
        self.downsample = settings.downsample
        self.clip = settings.normalise

    def data_vol_setup(self, data_vol):
        if self.downsample:
            data_vol = self.downsample_data(data_vol)
        if self.clip:
            data_vol = self.clip_to_uint8(data_vol)
        if np.isnan(data_vol).any():
            logging.info(f"Replacing NaN values.")
            data_vol = np.nan_to_num(data_vol, copy=False)
        return data_vol

    def load_in_vol(self, settings, data_path):
        load_path = Path(getattr(settings, data_path))
        nexus = load_path.suffix == ".nxs"
        return self.numpy_from_hdf5(
            load_path, hdf5_path=settings.hdf5_path, nexus=nexus
        )

    def downsample_data(self, data, factor=2):
        logging.info(f"Downsampling data by a factor of {factor}.")
        return block_reduce(data, block_size=(factor, factor, factor), func=np.nanmean)

    def numpy_from_hdf5(self, path, hdf5_path="/data", nexus=False):
        """Returns a numpy array when given a path to an HDF5 file.

        The data is assumed to be found in '/data' in the file.

        Args:
            path(pathlib.Path): The path to the HDF5 file.
            hdf5_path (str): The internal HDF5 path to the data.

        Returns:
            numpy.array: A numpy array object for the data stored in the HDF5 file.
        """

        with h5.File(path, "r") as f:
            if nexus:
                try:
                    data = f["processed/result/data"][()]
                except KeyError:
                    logging.error(
                        "NXS file: Couldn't find data at 'processed/result/data' trying another path."
                    )
                    try:
                        data = f["entry/final_result_tomo/data"][()]
                    except KeyError:
                        print(
                            "NXS file: Could not find entry at entry/final_result_tomo/data, exiting!"
                        )
                        sys.exit(1)
            else:
                data = f[hdf5_path][()]
        return data

    def clip_to_uint8(self, data):
        """Clips data to a certain number of st_devs of the mean and reduces
        bit depth to uint8.

        Args:
            data(np.array): The data to be processed.

        Returns:
            np.array: A unit8 data array.
        """
        logging.info("Clipping data and converting to uint8.")
        logging.info("Calculating mean of data...")
        data_mean = np.nanmean(data)
        logging.info(f"Mean {data_mean}. Calculating standard deviation.")
        data_st_dev = np.nanstd(data)
        logging.info(f"Std dev: {data_st_dev}. Calculating stats.")
        num_vox = data.size
        lower_bound = data_mean - (data_st_dev * self.st_dev_factor)
        upper_bound = data_mean + (data_st_dev * self.st_dev_factor)
        with np.errstate(invalid="ignore"):
            gt_ub = (data > upper_bound).sum()
            lt_lb = (data < lower_bound).sum()
        logging.info(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
        logging.info(
            f"Number of voxels above upper bound to be clipped {gt_ub} - percentage {gt_ub/num_vox * 100:.3f}%"
        )
        logging.info(
            f"Number of voxels below lower bound to be clipped {lt_lb} - percentage {lt_lb/num_vox * 100:.3f}%"
        )
        if np.isnan(data).any():
            logging.info(f"Replacing NaN values.")
            data = np.nan_to_num(data, copy=False, nan=data_mean)
        logging.info("Rescaling intensities.")
        if np.issubdtype(data.dtype, np.integer):
            logging.info(
                "Data is already in integer dtype, converting to float for rescaling."
            )
            data = data.astype(np.float)
        data = np.clip(data, lower_bound, upper_bound, out=data)
        data = np.subtract(data, lower_bound, out=data)
        data = np.divide(data, (upper_bound - lower_bound), out=data)
        data = np.clip(data, 0.0, 1.0, out=data)
        logging.info("Converting to uint8.")
        data = np.multiply(data, 255, out=data)
        return data.astype(np.uint8)

    def is_not_consecutive(self, num_list):
        maximum = max(num_list)
        if sum(num_list) == maximum * (maximum + 1) / 2:
            return False
        return True

    def fix_label_classes(self, data_vol, seg_classes):
        """Changes the data values of classes in a segmented volume so that
        they start from zero.

        Args:
            seg_classes(list): An ascending list of the labels in the volume.
        """
        for idx, current in enumerate(seg_classes):
            data_vol[data_vol == current] = idx
