#!/usr/bin/env python

import argparse
import logging
import os
from datetime import date
from pathlib import Path

from utilities import config as cfg
from utilities.cmdline import CheckExt
from utilities.data import (PredictionDataSlicer, SettingsData,
                            TrainingDataSlicer)
from utilities.unet2d import Unet2dPredictor, Unet2dTrainer


def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/training_data/file.h5] [path/to/segmentation/file.h5] [path/to/data_for_prediction/file.h5]",
        description="First, train a 2d U-net model on the 3d data and corresponding"
        " segmentation provided in the files. Then predict segmentation of a 3d data"
        " volume using the 2d model created."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(cfg.TRAIN_DATA_ARG, metavar='Path to training image data volume', type=str,
                        action=CheckExt(cfg.TRAIN_DATA_EXT),
                        help='the path to an HDF5 file containing the imaging data volume for training')
    parser.add_argument(cfg.LABEL_DATA_ARG, metavar='Path to label volume', type=str,
                        action=CheckExt(cfg.LABEL_DATA_EXT),
                        help='the path to an HDF5 file containing a segmented volume for training')
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
    train_settings_path = Path(root_path, cfg.SETTINGS_DIR,
                         cfg.TRAIN_SETTINGS_FN)
    predict_settings_path = Path(root_path, cfg.SETTINGS_DIR,
                               cfg.PREDICTION_SETTINGS_FN)
    train_settings = SettingsData(train_settings_path, args)
    predict_settings = SettingsData(predict_settings_path, args)
    # Select the requested GPU
    logging.info(f"Setting CUDA device {predict_settings.cuda_device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_settings.cuda_device)
    # Set up the TrainingDataSlicer and slice the data volumes into image files
    data_im_out_dir = root_path/train_settings.data_im_dirname  # dir for data imgs
    seg_im_out_dir = root_path/train_settings.seg_im_out_dirname  # dir for seg imgs
    train_slicer = TrainingDataSlicer(train_settings)
    train_slicer.output_data_slices(data_im_out_dir)
    train_slicer.output_label_slices(seg_im_out_dir)
    # Set up the UnetTrainer
    trainer = Unet2dTrainer(data_im_out_dir, seg_im_out_dir, train_slicer.codes,
                            train_settings)
    # Train the model
    trainer.train_model()
    # Save the model
    model_fn = f"{date.today()}_{train_settings.model_output_fn}"
    model_out = Path(root_path, model_fn)
    trainer.save_model_weights(model_out)
    # Save a figure showing the predictions
    trainer.output_prediction_figure(model_out)
    # Clean up all the saved slices
    train_slicer.clean_up_slices()
    # Prediction 
    predictor = Unet2dPredictor(root_path)
    predictor.get_model_from_trainer(trainer)
    # Create a slicer to slice and predict the segmentations from the data
    predict_slicer = PredictionDataSlicer(predict_settings, predictor)
    predict_slicer.predict_12_ways(root_path)
