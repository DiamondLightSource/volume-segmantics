#!/usr/bin/env python

import argparse
import logging
from datetime import date
from pathlib import Path
import sys

from utilities import config as cfg
from utilities.cmdline import CheckExt
from utilities.settingsdata import SettingsData
from utilities.slicers.trainingslicers import TrainingDataSlicer
from utilities.unet2d import Unet2dTrainer
from utilities.dataloaders import get_2d_dataloaders


def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s --data <path(s)/to/data/file(s)> --labels <path(s)/to/segmentation/file(s)> --data_dir path/to/data_directory",
        description="Train a 2d U-net model on the 3d data and corresponding"
        " segmentation provided in the files."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument("--" + cfg.TRAIN_DATA_ARG, metavar='Path(s) to training image data volume(s)', type=str,
                        action=CheckExt(cfg.TRAIN_DATA_EXT),
                        nargs="+", required=True,
                        help='the path(s) to file(s) containing the imaging data volume for training')
    parser.add_argument("--" + cfg.LABEL_DATA_ARG, metavar='Path(s) to label volume(s)', type=str,
                        action=CheckExt(cfg.LABEL_DATA_EXT),
                        nargs="+", required=True,
                        help='the path(s) to file(s) containing a segmented volume for training')
    parser.add_argument("--" + cfg.DATA_DIR_ARG, metavar='Path to settings and output directory (optional)', type=str,
                        nargs="?", default=Path.cwd(),
                        help='path to a directory containing the "unet-settings", data will be also be output to this location')
    return parser

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT,
        datefmt=cfg.LOGGING_DATE_FMT)
    # Parse args and check correct numer of volumes given
    parser = init_argparse()
    args = parser.parse_args()
    data_vols = getattr(args, cfg.TRAIN_DATA_ARG)
    label_vols = getattr(args, cfg.LABEL_DATA_ARG)
    root_path = Path(getattr(args, cfg.DATA_DIR_ARG)).resolve() 
    if len(data_vols) != len(label_vols):
        logging.error("Number of data volumes and number of label volumes must be equal!")
        sys.exit(1)
    # Create the settings object
    settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.TRAIN_SETTINGS_FN)
    settings = SettingsData(settings_path)
    data_im_out_dir = root_path/settings.data_im_dirname # dir for data imgs
    seg_im_out_dir = root_path/settings.seg_im_out_dirname # dir for seg imgs
    # Keep track of the number of labels
    max_label_no = 0
    label_codes = None
    # Set up the DataSlicer and slice the data volumes into image files
    for count, (data_vol_path, label_vol_path) in enumerate(zip(data_vols, label_vols)):
        slicer = TrainingDataSlicer(settings, data_vol_path, label_vol_path)
        data_prefix, label_prefix = f"data{count}", f"seg{count}"
        slicer.output_data_slices(data_im_out_dir, data_prefix)
        slicer.output_label_slices(seg_im_out_dir, label_prefix)
        if slicer.num_seg_classes > max_label_no:
            max_label_no = slicer.num_seg_classes
            label_codes = slicer.codes
    assert(label_codes is not None)
    # Set up the DataLoader to load in and augment the data
    train_loader, valid_loader = get_2d_dataloaders(data_im_out_dir, seg_im_out_dir, settings)
    # Set up the UnetTrainer
    trainer = Unet2dTrainer(data_im_out_dir, seg_im_out_dir, label_codes,
                            settings)
    # Train the model
    trainer.train_model()
    # Save the model
    model_fn = f"{date.today()}_{settings.model_output_fn}"
    model_out = Path(root_path, model_fn)
    trainer.save_model_weights(model_out)
    # Save a figure showing the predictions
    trainer.output_prediction_figure(model_out)
    # Clean up all the saved slices
    slicer.clean_up_slices()
