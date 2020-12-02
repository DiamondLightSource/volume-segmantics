#!/usr/bin/env python 

import argparse
import logging
import os

from pathlib import Path

from utilities.data import SettingsData, PredictionDataSlicer
from utilities.unet2d import Unet2dPredictor
from utilities.cmdline import CheckExt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/model/file.zip] [path/to/data/file.h5]",
        description="Predict segmentation of a 3d data volume using the 2d"
        "model provided."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('model_path', metavar='Model file path', type=str,
                        action=CheckExt({'zip'}),
                        help='the path to a zip file containing the model weights.')
    parser.add_argument('data_vol_path', metavar='Data volume file path', type=str,
                        action=CheckExt({'h5', 'hdf5'}),
                        help='the path to an HDF5 file containing the data to segment.')
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    root_path = Path.cwd()  # For module load script, use the CWD
    # Set up the settings
    parser = init_argparse()
    args = parser.parse_args()
    settings_path = Path(root_path, 'unet-settings',
                         '2d_unet_predict_settings.yaml')
    settings = SettingsData(settings_path, args)
    # Select the requsted GPU
    logging.info(f"Setting CUDA device {settings.cuda_device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings.cuda_device)
    # Create a model from the saved .zip file
    predictor = Unet2dPredictor(root_path)
    predictor.create_model_from_zip(settings.model_path)
    # Create a slicer to slice and predict the segmentations from the data
    slicer = PredictionDataSlicer(settings, predictor)
    slicer.predict_12_ways(root_path)
