#!/usr/bin/env python 

import argparse
import logging
import os
import warnings
from pathlib import Path

from utilities import config as cfg
from utilities.cmdline import CheckExt
from utilities.data import (PredictionDataSlicer, PredictionHDF5DataSlicer,
                            SettingsData)
from utilities.unet2d import Unet2dPredictor

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
        " model provided."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(cfg.MODEL_PTH_ARG, metavar='Model file path', type=str,
                        action=CheckExt(cfg.MODEL_DATA_EXT),
                        help='the path to a zip file containing the model weights.')
    parser.add_argument(cfg.PREDICT_DATA_ARG, metavar='Path to prediction data volume', type=str,
                        action=CheckExt(cfg.PREDICT_DATA_EXT),
                        help='the path to an HDF5 file containing the imaging data to segment')
    return parser

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT,
        datefmt=cfg.LOGGING_DATE_FMT)
    root_path = Path.cwd()  # For module load script, use the CWD
    # Set up the settings
    parser = init_argparse()
    args = parser.parse_args()
    settings_path = Path(root_path, cfg.SETTINGS_DIR,
                         cfg.PREDICTION_SETTINGS_FN)
    settings = SettingsData(settings_path, args)
    # Select the requested GPU
    logging.info(f"Setting CUDA device {settings.cuda_device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings.cuda_device)
    # Create a model from the saved .zip file
    model_file_path = getattr(settings, cfg.MODEL_PTH_ARG)
    predictor = Unet2dPredictor(root_path)
    predictor.create_model_from_zip(model_file_path)
    # Create a slicer to slice and predict the segmentations from the data
    if settings.use_max_probs:
        slicer = PredictionHDF5DataSlicer(settings, predictor)
    else:
        slicer = PredictionDataSlicer(settings, predictor)
    slicer.predict_12_ways(root_path)
