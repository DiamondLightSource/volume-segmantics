import logging
import pathlib
import sys
from enum import Enum
from itertools import chain, product
from types import SimpleNamespace
from typing import List, Union, Tuple

import h5py as h5
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as F
from skimage.measure import block_reduce

import volume_segmantics.utilities.config as cfg
from types import SimpleNamespace
from pathlib import Path


class Quality(Enum):
    """An Enum that holds values describing the quality of segmentation
    prediction required.

    Refers to the number of axes/rotations that the segmentation is predicted
    in. e.g. Low quality, single axis (x, y) prediction; medium quality, three axis (x, y),
    (x, z), (y, z) prediction; high quality 12 way (3 axis and 4 rotations) prediction.
    """

    LOW = 1
    MEDIUM = 3
    HIGH = 12


class Axis(Enum):
    Z = 0
    Y = 1
    X = 2
    ALL = 4


class ModelType(Enum):
    U_NET = 1
    U_NET_PLUS_PLUS = 2
    FPN = 3
    DEEPLABV3 = 4
    DEEPLABV3_PLUS = 5
    MA_NET = 6
    LINKNET = 7
    PAN = 8


def create_enum_from_setting(setting_str, enum):
    if isinstance(setting_str, Enum):
        return setting_str
    try:
        output_enum = enum[setting_str.upper()]
    except KeyError:
        options = [k.name for k in enum]
        logging.error(
            f"{enum.__name__}: {setting_str} is not valid. Options are {options}."
        )
        sys.exit(1)
    return output_enum


def get_prediction_quality(settings: SimpleNamespace) -> Enum:
    pred_quality = create_enum_from_setting(settings.quality, Quality)
    return pred_quality


def get_model_type(settings: SimpleNamespace) -> Enum:
    model_type = create_enum_from_setting(settings.model["type"], ModelType)
    return model_type


def get_training_axis(settings: SimpleNamespace) -> Enum:
    try:
        axis_setting = settings.training_axes
    except AttributeError:
        axis_setting = "All"
    axis_enum = create_enum_from_setting(axis_setting, Axis)
    return axis_enum


def get_prediction_axis(settings: SimpleNamespace) -> Enum:
    try:
        axis_setting = settings.prediction_axis
    except AttributeError:
        axis_setting = "Z"
    axis_enum = create_enum_from_setting(axis_setting, Axis)
    return axis_enum


def setup_path_if_exists(input_param):
    if isinstance(input_param, str):
        return Path(input_param)
    elif isinstance(input_param, Path):
        return input_param
    else:
        return None


def get_batch_size(settings: SimpleNamespace, prediction: bool = False) -> int:

    cuda_device_num = settings.cuda_device
    total_gpu_mem = torch.cuda.get_device_properties(cuda_device_num).total_memory
    allocated_gpu_mem = torch.cuda.memory_allocated(cuda_device_num)
    free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024**3

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


def crop_tensor_to_array(tensor: torch.Tensor, yx_dims: List[int]) -> np.array:
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = F.center_crop(tensor, yx_dims)
    return tensor.detach().numpy()


def rotate_array_to_axis(array: np.array, axis: Axis = Axis.Z) -> np.array:
    if axis == Axis.Z:
        return array
    if axis == Axis.Y:
        return array.swapaxes(0, 1)
    if axis == Axis.X:
        return array.swapaxes(0, 2)


def one_hot_encode_array(input_array: np.array, num_labels: int) -> np.array:
    """Modified from https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy"""
    ncols = num_labels
    out = np.zeros((ncols, input_array.size), dtype=np.uint8)
    out[input_array.ravel(), np.arange(input_array.size)] = 1
    out.shape = (ncols,) + input_array.shape
    return out


def prepare_training_batch(
    batch: "list[torch.Tensor]", device: int, num_labels: int
) -> "tuple[torch.Tensor, torch.Tensor]":
    inputs = batch[0].to(device)
    targets = batch[1].to(torch.int64)
    # One hot encode the channels
    targets = torch.nn.functional.one_hot(targets, num_classes=num_labels)
    targets = targets.permute((0, 3, 1, 2)).to(device, dtype=torch.uint8)
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
    """Returns a numpy array  and chunking info when given a path
    to an HDF5 file.

    The data is assumed to be found in '/data' in the file.

    Args:
        path(pathlib.Path): The path to the HDF5 file.
        hdf5_path (str): The internal HDF5 path to the data.

    Returns:
        tuple(numpy.array, tuple(int, int)) : A numpy array
        for the data and a tuple with the chunking size.
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
    input_data_chunking = dataset.chunks
    return dataset[()], input_data_chunking


def get_numpy_from_path(
    path: pathlib.Path, internal_path: str = "/data"
) -> Tuple[np.array, Union[Tuple[int, int], bool]]:
    """Helper function that returns numpy array and chunking(if used)
    according to file extension.

    Args:
        path (pathlib.Path): Path to file
        internal_path (str, optional): Path inside HDF5 file. Defaults to "/data".

    Returns:
        Tuple[np.array, Union[Tuple[int, int], bool]]: Tuple with data array and
        either chunking tuple, or True.
    """
    if path.suffix in cfg.TIFF_SUFFIXES:
        return numpy_from_tiff(path), True
    elif path.suffix in cfg.HDF5_SUFFIXES:
        nexus = path.suffix == ".nxs"
        return numpy_from_hdf5(path, hdf5_path=internal_path, nexus=nexus)


def sequential_labels(unique_labels: np.array) -> bool:
    if not np.where(np.diff(unique_labels) != 1)[0].size:
        return True
    else:
        return False


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
        data = data.astype(float)
    data = np.clip(data, lower_bound, upper_bound, out=data)
    data = np.subtract(data, lower_bound, out=data)
    data = np.divide(data, (upper_bound - lower_bound), out=data)
    # data = (data - lower_bound) / (upper_bound - lower_bound)
    data = np.clip(data, 0.0, 1.0, out=data)
    # data = exposure.rescale_intensity(data, in_range=(lower_bound, upper_bound))
    logging.info("Converting to uint8.")
    data = np.multiply(data, 255, out=data)
    return data.astype(np.uint8)


def get_num_of_ims(vol_shape: Tuple, axis_enum: Axis):
    """Calculates the number of images that will be created when slicing
    an image volume in specified planes or combinations of planes.

    Args:
        vol_shape (tuple): 3d volume shape (z, y, x).
        axis_enum (Enum): Defines specific axis or all axes

    Returns:
        int: Total number of images that will be created when the volume is
        sliced.
    """
    if axis_enum == Axis.ALL:
        return sum(vol_shape)
    else:
        return vol_shape[axis_enum.value]


def get_axis_index_pairs(vol_shape: Tuple, axis_enum: Axis):
    """Gets all combinations of axis and image slice indices that are found
    in a 3d volume.

    Args:
        vol_shape (tuple): 3d volume shape (z, y, x)
        axis_enum (Enum): Defines specific axis or all axes

    Returns:
        itertools.chain: An iterable containing all combinations of axis
        and image index that are found in the volume.
    """

    if axis_enum == Axis.ALL:
        return chain(
            product("z", range(vol_shape[0])),
            product("y", range(vol_shape[1])),
            product("x", range(vol_shape[2])),
        )
    else:
        return product(axis_enum.name.lower(), range(vol_shape[axis_enum.value]))


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
