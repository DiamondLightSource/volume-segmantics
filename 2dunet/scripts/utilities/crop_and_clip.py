#!/usr/bin/env python
import argparse
import os.path
import logging
from pathlib import Path
import h5py as h5
import sys
import numpy as np
from skimage.measure import block_reduce
from skimage import img_as_ubyte, exposure

INPUT_FILE_EXT = {"h5", "hdf5", "nxs"}
# Logging format
LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"
STDEV_FACTOR = 2.575
DEFAULT_SIZE = 256
DEFAULT_OFFSET = [0, 0, 0]

def CheckExt(choices):
    """Wrapper to return the class
    """
    class Act(argparse.Action):
        """Class to allow checking of filename extensions in argparse. Taken
        from https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
        """
        def __call__(self, parser, namespace, fname, option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(
                    option_string) if option_string else ''
                parser.error("Wrong filetype: file doesn't end with {}{}".format(
                    choices, option_string))
            else:
                setattr(namespace, self.dest, fname)

    return Act

def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/3d/image/data/file] options...",
        description="Cuts out and saves a smaller 3d volume from a larger 3d"\
        " imaging volume. Data can be downsampled and/or intensities clipped"\
        " followed by changing bit depth to uint8."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument("input_file_path", metavar='Input file path', type=str,
                        action=CheckExt(INPUT_FILE_EXT),
                        help='the path to a file containing 3d image data.')
    parser.add_argument('--size', nargs=1, type=int,
                        help='the size of the cube (default is 256).')
    parser.add_argument('--offset', nargs=3, type=int,
                        help='three values specifying the Z Y and X offset from the centre.')
    parser.add_argument('--downsample', nargs=1, type=int,
                        help='a factor to downsample the data by.')
    parser.add_argument('--clip',
                        action='store_true',
                        help="if specified the image intensities will be clipped, "
                        "rescaled and reduced to uint8 bit depth.")
    parser.add_argument('--coords',nargs=6, type=int,
                        help="specify coordinates for cube in the form: zstart, zend,"
                        " ystart, yend, xstart, xend. Cannot be specfied in conjunction with"
                        " --size and/or--offset")
    return parser

def numpy_from_hdf5(path, hdf5_path='/data', nexus=False, lazy=False):
    """Returns a numpy array when given a path to an HDF5 file.

    The data is assumed to be found in '/data' in the file.

    Args:
        path(pathlib.Path): The path to the HDF5 file.
        hdf5_path (str): The internal HDF5 path to the data.

    Returns:
        numpy.array: A numpy array object for the data stored in the HDF5 file.
    """
    
    f = h5.File(path, 'r')
    if nexus:
        try:
            data = f['processed/result/data']
        except KeyError:
            logging.error("NXS file: Couldn't find data at 'processed/result/data' trying another path.")
            try:
                data = f['entry/final_result_tomo/data']
            except KeyError:
                logging.error("NXS file: Could not find entry at entry/final_result_tomo/data, exiting!")
                sys.exit(1)
    else:
        data = f[hdf5_path]
    if not lazy:
        data = data[()]
        f.close()
        return data
    return data, f

def clip_to_uint8(data):
        """Clips data to a certain number of st_devs of the mean and reduces
        bit depth to uint8.

        Args:
            data(np.array): The data to be processed.

        Returns:
            np.array: A unit8 data array.
        """
        logging.info("Clipping data and converting to uint8.")
        logging.info("Calculating mean intensity:")
        data_mean = np.nanmean(data)
        logging.info(f"Mean value: {data_mean}. Calculating standard deviation.")
        # diff_mat = np.ravel(data - data_mean)
        # not_nan_size = data.size - np.isnan(diff_mat).sum()
        # data_st_dev = np.sqrt(np.nansum(diff_mat * diff_mat)/not_nan_size)
        data_st_dev = np.nanstd(data)
        data_size = data.size
        lower_bound = data_mean - (data_st_dev * STDEV_FACTOR)
        upper_bound = data_mean + (data_st_dev * STDEV_FACTOR)
        logging.info(f"Std dev: {data_st_dev}. Calculating stats.")
        with np.errstate(invalid='ignore'):
            gt_ub = (data > upper_bound).sum()
            lt_lb = (data < lower_bound).sum()
        logging.info(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
        logging.info(
            f"Number of voxels above upper bound to be clipped {gt_ub} - percentage {gt_ub/data_size * 100:.3f}%")
        logging.info(
            f"Number of voxels below lower bound to be clipped {lt_lb} - percentage {lt_lb/data_size * 100:.3f}%")
        if np.isnan(data).any():
            logging.info(f"Replacing NaN values.")
            data = np.nan_to_num(data, copy=False, nan=data_mean)
        logging.info("Rescaling intensities.")
        data = np.clip(data, lower_bound, upper_bound, out=data)
        data = np.subtract(data, lower_bound, out=data)
        data = np.divide(data, (upper_bound - lower_bound), out=data)
        data = np.clip(data, 0.0, 1.0, out=data)
        logging.info("Converting to uint8.")
        data = np.multiply(data, 255, out=data)
        return data.astype(np.uint8)

def check_coords(coords, data_shape, downsample):
    logging.info("Checking coordinates.")
    coords = np.array(coords)
    if any(coords < 0):
        logging.error("Can not have negative coords! Exiting")
        sys.exit(1)
    if not downsample:
        downsample = 1
    else:
        downsample = downsample[0]
    bounding_box = np.array(data_shape) / float(downsample)
    if any(coords[1::2] > bounding_box):
        logging.error("Coordinates can not be outside final volume size! Exiting")
        sys.exit(1)

def save_data_subvolume(root_path, data, cube_size, offset, coords):
    if coords:
        dims = {'z': (coords[0], coords[1]),
                'y': (coords[2], coords[3]),
                'x': (coords[4], coords[5])}
    else:
        centre = np.array(data.shape) // 2
        half_edge = cube_size // 2
        z_min = centre[0] - half_edge + offset[0]
        y_min = centre[1] - half_edge + offset[1]
        x_min = centre[2] - half_edge + offset[2]
        dims = {'z': (z_min, z_min + cube_size),
                'y': (y_min, y_min + cube_size),
                'x': (x_min, x_min + cube_size)}
        coords = np.array([a for b in dims.values() for a in b])
        if any(coords < 0):
            logging.error("Can not have negative coords! Exiting")
            sys.exit(1)
        if any(coords[1::2] > np.array(data.shape)):
            logging.error("Coordinates can not be outside final volume size! Exiting")
            sys.exit(1)
    dim_string = f"{'_'.join(['-'.join(map(str, x)) for x in (dims['z'], dims['y'], dims['x'])])}"
    data_out_fname = f"subvolume_{dim_string}.h5"
    logging.info(f"Saving data out to {root_path/data_out_fname}.")
    with h5.File(root_path/data_out_fname, 'w') as f:
        f["/data"] = data[slice(*dims['z']),
                                  slice(*dims['y']), slice(*dims['x'])]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=LOGGING_FMT,
        datefmt=LOGGING_DATE_FMT)
    root_path = Path.cwd()
    data = None
    # Extractinput args #
    parser = init_argparse()
    args = vars(parser.parse_args())
    input_path = Path(args["input_file_path"])
    nexus_flag = input_path.suffix == ".nxs"
    cube_size = args["size"]
    coords = args['coords']
    offset = args['offset']
    clip = args['clip']
    downsample = args['downsample']
    # Check input args 
    if any([(cube_size and coords), (coords and offset)]):
        logging.error("Invalid combination of arguments!")
        sys.exit(1)
    if not input_path.exists():
        logging.error(f"Input file: {input_path} does not exist. Exiting!")
        sys.exit(1)
    data_vol, f = numpy_from_hdf5(input_path, nexus=nexus_flag, lazy=True)
    data_shape = data_vol.shape
    if coords is not None:
        check_coords(coords,data_shape, downsample)
    # Create default values 
    if not cube_size:
        cube_size = DEFAULT_SIZE
    else:
        cube_size = cube_size[0]
    if not offset:
        offset = DEFAULT_OFFSET
    # Load entire volume for clipping or downsampling 
    if clip or downsample:
        logging.info("Loading in data:")
        data = data_vol[()]
        f.close()

    if downsample is not None:
        down_factor = downsample[0]
        logging.info(f"Downsampling by a factor of {down_factor}.")
        block_size = (down_factor,)*3
        data = block_reduce(data, block_size=block_size, func=np.nanmean)
    
    if clip:
        data = clip_to_uint8(data)

    if data is None:
        data = data_vol
    save_data_subvolume(root_path, data, cube_size, offset, coords)

    