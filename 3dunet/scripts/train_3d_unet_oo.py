#!/usr/bin/env python

import argparse
import os
import logging
from datetime import date
from pathlib import Path

from utilities import config as cfg
from utilities.cmdline import CheckExt
from utilities.data_utils_base import SettingsData
from utilities.data_utils_3d import TrainingData3dSampler
from utilities.unet3d import Unet3dTrainer


def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/data/file.h5] [path/to/segmentation/file.h5]"
        "[path/to/validation_data/file.h5] [path/to/validation_segmentation/file.h5]",
        description="Train a 3d U-net model on the 3d data and corresponding"
        " segmentation provided in the files.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        cfg.TRAIN_DATA_ARG,
        metavar="Path to training image data volume",
        type=str,
        action=CheckExt(cfg.TRAIN_DATA_EXT),
        help="the path to an HDF5 file containing the imaging data volume for training",
    )
    parser.add_argument(
        cfg.LABEL_DATA_ARG,
        metavar="Path to label volume",
        type=str,
        action=CheckExt(cfg.LABEL_DATA_EXT),
        help="the path to an HDF5 file containing a segmented volume for training",
    )
    parser.add_argument(
        cfg.TRAIN_VAL_DATA_ARG,
        metavar="Path to training image data volume",
        type=str,
        action=CheckExt(cfg.TRAIN_DATA_EXT),
        help="the path to an HDF5 file containing the imaging data validation volume for training",
    )
    parser.add_argument(
        cfg.LABEL_VAL_DATA_ARG,
        metavar="Path to label volume",
        type=str,
        action=CheckExt(cfg.LABEL_DATA_EXT),
        help="the path to an HDF5 file containing a segmented validation volume for training",
    )
    parser.add_argument(cfg.DATA_DIR_ARG, metavar='Path to settings and output directory',
                        type=str,
                        help='the path to a directory containing the "unet-settings", data will be output to this location')
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
    )
    # Set up the settings
    parser = init_argparse()
    args = parser.parse_args()
    root_path = Path(getattr(args, cfg.DATA_DIR_ARG)).resolve()
    settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.TRAIN_SETTINGS_3D)
    settings = SettingsData(settings_path, args)
    # Set the CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_device)
    # Set up the DataSampler to load in the data
    sampler = TrainingData3dSampler(settings)
    # Set up the UnetTrainer
    trainer = Unet3dTrainer(sampler, settings)
    # # Train the model
    model_fn = f"{date.today()}_{settings.model_output_fn}.pytorch"
    model_out = Path(root_path, model_fn)
    trainer.train_model(model_out, settings.num_epochs, settings.patience)
    trainer.output_loss_fig(model_out)
    # Predict a segmentation for the validation region
    valid_out_fn = f"{date.today()}_{settings.model_output_fn}_validation_prediction.h5"
    valid_out_pth = Path(root_path, valid_out_fn)
    validation_prediction = sampler.predict_volume(trainer.model, sampler.data_val_vol, valid_out_pth)
    # Output a figure with data and prediction images
    fig_out_fn = f"{date.today()}_{settings.model_output_fn}_prediction_output.png"
    fig_out_pth = Path(root_path, fig_out_fn)
    sampler.plot_predict_figure(sampler.seg_val_vol, validation_prediction, fig_out_pth, validation=True)

