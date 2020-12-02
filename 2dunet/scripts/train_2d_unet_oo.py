#!/usr/bin/env python

import argparse
import logging
from datetime import date
from pathlib import Path

from utilities.data import TrainingDataSlicer, SettingsData
from utilities.unet2d import Unet2dTrainer
from utilities.cmdline import CheckExt


def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/data/file.h5] [path/to/segmentation/file.h5]",
        description="Train a 2d U-net model on the 3d data and corresponding"
        "segmentation provided in the files."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('data_vol_path', metavar='Image data file path', type=str,
                        action=CheckExt({'h5', 'hdf5'}),
                        help='the path to an HDF5 file containing the imaging data volume.')
    parser.add_argument('seg_vol_path', metavar='Segmentation file path', type=str,
                        action=CheckExt({'h5', 'hdf5'}),
                        help='the path to an HDF5 file containing a segmented volume.')
    return parser

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    root_path = Path.cwd()  # For module load script, use the CWD
    # Set up the settings
    parser = init_argparse()
    args = parser.parse_args()
    settings_path = Path(root_path, 'unet-settings', '2d_unet_train_settings.yaml')
    settings = SettingsData(settings_path, args)
    # Set up the DataSlicer and slice the data volumes into image files
    data_im_out_dir = root_path/settings.data_im_dirname # dir for data imgs
    seg_im_out_dir = root_path/settings.seg_im_out_dirname # dir for seg imgs
    slicer = TrainingDataSlicer(settings)
    slicer.output_data_slices(data_im_out_dir)
    slicer.output_label_slices(seg_im_out_dir)
    # Set up the UnetTrainer
    trainer = Unet2dTrainer(data_im_out_dir, seg_im_out_dir, slicer.codes,
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
