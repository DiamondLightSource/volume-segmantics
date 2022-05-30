#!/usr/bin/env python 

import argparse
import logging
import warnings
from datetime import date
from pathlib import Path

from utilities import config as cfg
from utilities.arg_parsing import CheckExt
from data.settings_data import SettingsData
from unet2d.prediction_manager import Unet2DPredictionManager
from unet2d.predictor import Unet2dPredictor

warnings.filterwarnings("ignore", category=UserWarning)

def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s path/to/model/file.zip path/to/data/file [path/to/data_directory]",
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
    parser.add_argument("--" + cfg.DATA_DIR_ARG, metavar='Path to settings and output directory (optional)', type=str,
                        nargs="?", default=Path.cwd(),
                        help='path to a directory containing the "unet-settings", data will be also be output to this location')
    return parser

def create_output_path(root_path, data_vol_path):
    pred_out_fn = f"{date.today()}_{data_vol_path.stem}_2dUnet_vol_pred.h5"
    return Path(root_path, pred_out_fn)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT,
        datefmt=cfg.LOGGING_DATE_FMT)
    parser = init_argparse()
    args = parser.parse_args()
    root_path = Path(getattr(args, cfg.DATA_DIR_ARG)).resolve()
    settings_path = Path(root_path, cfg.SETTINGS_DIR,
                         cfg.PREDICTION_SETTINGS_FN)
    settings = SettingsData(settings_path)
    model_file_path = getattr(args, cfg.MODEL_PTH_ARG)
    predictor = Unet2dPredictor(model_file_path, settings)
    data_vol_path = Path(getattr(args, cfg.PREDICT_DATA_ARG))
    output_path = create_output_path(root_path, data_vol_path)
    pred_manager = Unet2DPredictionManager(predictor, data_vol_path, settings)
    pred_manager.predict_volume_to_path(output_path)
