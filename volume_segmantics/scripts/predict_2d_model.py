#!/usr/bin/env python

import logging
import warnings
from datetime import date
from pathlib import Path

import volume_segmantics.utilities.config as cfg
from volume_segmantics.data import get_settings_data
from volume_segmantics.model import VolSeg2DPredictionManager
from volume_segmantics.utilities import get_2d_prediction_parser

warnings.filterwarnings("ignore", category=UserWarning)


def create_output_path(root_path, data_vol_path):
    pred_out_fn = f"{date.today()}_{data_vol_path.stem}_2d_model_vol_pred.h5"
    return Path(root_path, pred_out_fn)


def main():
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
    )
    # Parse Args
    parser = get_2d_prediction_parser()
    args = parser.parse_args()
    # Define paths
    root_path = Path(getattr(args, cfg.DATA_DIR_ARG)).resolve()
    settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.PREDICTION_SETTINGS_FN)
    model_file_path = getattr(args, cfg.MODEL_PTH_ARG)
    data_vol_path = Path(getattr(args, cfg.PREDICT_DATA_ARG))
    output_path = create_output_path(root_path, data_vol_path)
    # Get settings object
    settings = get_settings_data(settings_path)
    # Create prediction manager and predict
    pred_manager = VolSeg2DPredictionManager(model_file_path, data_vol_path, settings)
    pred_manager.predict_volume_to_path(output_path)


if __name__ == "__main__":
    main()
