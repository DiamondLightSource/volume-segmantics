#!/usr/bin/env python

import argparse
import os
import logging
from datetime import date
from pathlib import Path

from utilities import config as cfg
from utilities.cmdline import CheckExt
from utilities.data_utils_base import SettingsData
from utilities.data_utils_3d import PredictionData3dSampler
from utilities.unet3d import Unet3dPredictor


def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s path/to/model/file.pytorch path/to/data/file.h5 path/to/data_directory",
        description="Predict segmentation of a 3d data volume using the 3d"
        " model provided."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(cfg.MODEL_PTH_ARG, metavar='Model file path', type=str,
                        action=CheckExt(cfg.MODEL_DATA_EXT),
                        help='the path to a pytorch file containing the model weights.')
    parser.add_argument(cfg.PREDICT_DATA_ARG, metavar='Path to prediction data volume', type=str,
                        action=CheckExt(cfg.PREDICT_DATA_EXT),
                        help='the path to an HDF5 file containing the imaging data to segment')
    parser.add_argument("--" + cfg.DATA_DIR_ARG, metavar='Path to settings and output directory (optional)', type=str,
                        nargs="?", default=Path.cwd(),
                        help='path to a directory containing the "unet-settings", data will be also be output to this location')
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
    )
    # Set up the settings
    parser = init_argparse()
    args = parser.parse_args()
    root_path = Path(getattr(args, cfg.DATA_DIR_ARG)).resolve()
    settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.PREDICTION_SETTINGS_3D)
    settings = SettingsData(settings_path, args)
    # Set the CUDA device
    logging.info(f"Setting CUDA device to {settings.cuda_device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_device)
    # Create a model from the saved .pytorch file
    model_file_path = getattr(settings, cfg.MODEL_PTH_ARG)
    predictor = Unet3dPredictor()
    predictor.create_model_from_file(model_file_path)
    # Create a sampler to sample the 3d prediction volume
    sampler = PredictionData3dSampler(settings, predictor)
    # Predict a segmentation for the volume
    input_data_pth = Path(getattr(settings,cfg.PREDICT_DATA_ARG))
    pred_out_fn = f"{date.today()}_{input_data_pth.stem}_3dUnet_vol_pred.h5"
    pred_out_pth = Path(root_path, pred_out_fn)
    pred_out_vol = sampler.predict_volume(predictor.model, sampler.data_vol, pred_out_pth)
    # Output a figure with data and prediction images
    fig_out_fn = f"{date.today()}_{input_data_pth.stem}_prediction_output.png"
    fig_out_pth = Path(root_path, fig_out_fn)
    sampler.plot_predict_figure(sampler.data_vol, pred_out_vol, fig_out_pth)
