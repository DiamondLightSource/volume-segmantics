import logging
import sys
from enum import Enum
from itertools import chain, product

import h5py as h5
import imageio
import numpy as np
import torch
from skimage.measure import block_reduce

from utilities import config as cfg
from utilities.settingsdata import SettingsData


class Quality(Enum):
    LOW = 1
    MEDIUM = 3
    HIGH = 12


def get_batch_size(settings: SettingsData, prediction: bool = False) -> int:

    cuda_device_num = settings.cuda_device
    total_gpu_mem = torch.cuda.get_device_properties(cuda_device_num).total_memory
    allocated_gpu_mem = torch.cuda.memory_allocated(cuda_device_num)
    free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024 ** 3

    if free_gpu_mem < cfg.BIG_CUDA_THRESHOLD:
        batch_size = cfg.SMALL_CUDA_BATCH
    elif not prediction:
        batch_size = cfg.BIG_CUDA_TRAIN_BATCH
    else:
        batch_size = cfg.BIG_CUDA_PRED_BATCH
    
    logging.info(
        f"Free GPU memory is {free_gpu_mem:0.2f} GB. Batch size will be "
        f"{batch_size}."
    )
    return batch_size


def prepare_training_batch(batch: "list[torch.tensor]", device: int, num_labels):
    inputs = batch[0].to(device)
    targets = batch[1]
    # One hot encode the channels
    # TODO Replace with torch.nn.functional.one_hot
    channels = []
    for label_num in range(num_labels):
        channel = torch.zeros_like(targets)
        channel[targets == label_num] = 1
        channels.append(channel)
    targets = torch.stack(channels, 1).to(device, dtype=torch.uint8)
    return inputs, targets


def downsample_data(data, factor=2):
    logging.info(f"Downsampling data by a factor of {factor}.")
    return block_reduce(data, block_size=(factor, factor, factor), func=np.nanmean)


def numpy_from_tiff(path):
    """Returns a numpy array when given a path to an multipage TIFF file.

    Args:
        path(pathlib.Path): The path to the TIFF file.

    Returns:
        numpy.array: A numpy array object for the data stored in the TIFF file.
    """

    return imageio.volread(path)


def numpy_from_hdf5(path, hdf5_path="/data", nexus=False):
    """Returns a numpy array when given a path to an HDF5 file.

    The data is assumed to be found in '/data' in the file.

    Args:
        path(pathlib.Path): The path to the HDF5 file.
        hdf5_path (str): The internal HDF5 path to the data.

    Returns:
        numpy.array: A numpy array object for the data stored in the HDF5 file.
    """

    data_handle = h5.File(path, "r")
    if nexus:
        try:
            dataset = data_handle["processed/result/data"]
        except KeyError:
            logging.error(
                "NXS file: Couldn't find data at 'processed/result/data' trying another path."
            )
            try:
                dataset = data_handle["entry/final_result_tomo/data"]
            except KeyError:
                logging.error(
                    "NXS file: Could not find entry at entry/final_result_tomo/data, exiting!"
                )
                sys.exit(1)
    else:
        dataset = data_handle[hdf5_path]
    # self.input_data_chunking = dataset.chunks
    return dataset[()]


def get_numpy_from_path(path, internal_path="/data"):
    """Helper function that returns numpy array according to file extension.

    Args:
        path (pathlib.Path): The path to the data file.
        internal_path (str, optional): Internal path within HDF5 file. Defaults to "/data".

    Returns:
        numpy.ndarray: Numpy array from the file given in the path.
    """
    if path.suffix in cfg.TIFF_SUFFIXES:
        return numpy_from_tiff(path)
    elif path.suffix in cfg.HDF5_SUFFIXES:
        nexus = path.suffix == ".nxs"
        return numpy_from_hdf5(path, hdf5_path=internal_path, nexus=nexus)


def clip_to_uint8(data: np.array, data_mean: float, st_dev_factor: float) -> np.array:
    """Clips data to a certain number of st_devs of the mean and reduces
    bit depth to uint8.

    Args:
        data(np.array): The data to be processed.

    Returns:
        np.array: A unit8 data array.
    """
    logging.info("Clipping data and converting to uint8.")
    logging.info(f"Calculating standard deviation.")
    data_st_dev = np.nanstd(data)
    logging.info(f"Std dev: {data_st_dev}. Calculating stats.")
    num_vox = data.size
    lower_bound = data_mean - (data_st_dev * st_dev_factor)
    upper_bound = data_mean + (data_st_dev * st_dev_factor)
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
    # data = (data - lower_bound) / (upper_bound - lower_bound)
    data = np.clip(data, 0.0, 1.0, out=data)
    # data = exposure.rescale_intensity(data, in_range=(lower_bound, upper_bound))
    logging.info("Converting to uint8.")
    data = np.multiply(data, 255, out=data)
    return data.astype(np.uint8)


def get_num_of_ims(vol_shape):
    """Calculates the total number of images that will be created when slicing
    an image volume in the z, y and x planes.

    Args:
        vol_shape (tuple): 3d volume shape (z, y, x).

    Returns:
        int: Total number of images that will be created when the volume is
        sliced.
    """
    return sum(vol_shape)


def get_axis_index_pairs(vol_shape):
    """Gets all combinations of axis and image slice index that are found
    in a 3d volume.

    Args:
        vol_shape (tuple): 3d volume shape (z, y, x)

    Returns:
        itertools.chain: An iterable containing all combinations of axis
        and image index that are found in the volume.
    """
    return chain(
        product("z", range(vol_shape[0])),
        product("y", range(vol_shape[1])),
        product("x", range(vol_shape[2])),
    )


def axis_index_to_slice(vol, axis, index):
    """Converts an axis and image slice index for a 3d volume into a 2d
    data array (slice).

    Args:
        vol (3d array): The data volume to be sliced.
        axis (str): One of 'z', 'y' and 'x'.
        index (int): An image slice index found in that axis.

    Returns:
        2d array: A 2d image slice corresponding to the axis and index.
    """
    if axis == "z":
        return vol[index, :, :]
    if axis == "y":
        return vol[:, index, :]
    if axis == "x":
        return vol[:, :, index]


def save_data_to_hdf5(data, file_path, internal_path="/data", chunking=True):
    logging.info(f"Saving data of shape {data.shape} to {file_path}.")
    with h5.File(file_path, "w") as f:
        f.create_dataset(
            internal_path, data=data, chunks=chunking, compression=cfg.HDF5_COMPRESSION
        )
