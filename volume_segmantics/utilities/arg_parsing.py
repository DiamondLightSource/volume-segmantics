import argparse
from pathlib import Path

import volume_segmantics.utilities.config as cfg


def CheckExt(choices):
    """Wrapper to return the class"""

    class Act(argparse.Action):
        """Class to allow checking of filename extensions in argparse. Also
        checks whether file exists. Adapted from
        https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
        """

        def __call__(self, parser, namespace, fnames, option_string=None):
            # Modified to take in a list of filenames
            if isinstance(fnames, list):
                for fname in fnames:
                    self.check_path(parser, fname)
            else:
                self.check_path(parser, fnames)

            # If all okay, set attribute
            setattr(namespace, self.dest, fnames)

        def check_path(self, parser, fname):
            fname = Path(fname)
            ext = fname.suffix
            if ext not in choices:
                parser.error(f"Wrong filetype: file {fname} doesn't end with {choices}")
                # Check that file exists
            if not fname.is_file():
                parser.error(f"The file {str(fname)} does not appear to exist.")

    return Act


def get_2d_training_parser() -> argparse.ArgumentParser:
    """Argument parser for scripts that train a 2d network on a 3d volume.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s --data <path(s)/to/data/file(s)> --labels <path(s)/to/segmentation/file(s)> --data_dir path/to/data_directory",
        description="Train a 2d model on the 3d data and corresponding"
        " segmentation provided in the files.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "--" + cfg.TRAIN_DATA_ARG,
        metavar="Path(s) to training image data volume(s)",
        type=str,
        action=CheckExt(cfg.TRAIN_DATA_EXT),
        nargs="+",
        required=True,
        help="the path(s) to file(s) containing the imaging data volume for training",
    )
    parser.add_argument(
        "--" + cfg.LABEL_DATA_ARG,
        metavar="Path(s) to label volume(s)",
        type=str,
        action=CheckExt(cfg.LABEL_DATA_EXT),
        nargs="+",
        required=True,
        help="the path(s) to file(s) containing a segmented volume for training",
    )
    parser.add_argument(
        "--" + cfg.DATA_DIR_ARG,
        metavar="Path to settings and output directory (optional)",
        type=str,
        nargs="?",
        default=Path.cwd(),
        help='path to a directory containing the "volseg-settings", data will be also be output to this location',
    )
    return parser


def get_2d_prediction_parser() -> argparse.ArgumentParser:
    """Argument parser for scripts that use a 2d network to predict segmenation for a 3d volume.

    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s path/to/model/file.zip path/to/data/file [path/to/data_directory]",
        description="Predict segmentation of a 3d data volume using the 2d"
        " model provided.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        cfg.MODEL_PTH_ARG,
        metavar="Model file path",
        type=str,
        action=CheckExt(cfg.MODEL_DATA_EXT),
        help="the path to a zip file containing the model weights.",
    )
    parser.add_argument(
        cfg.PREDICT_DATA_ARG,
        metavar="Path to prediction data volume",
        type=str,
        action=CheckExt(cfg.PREDICT_DATA_EXT),
        help="the path to an HDF5 file containing the imaging data to segment",
    )
    parser.add_argument(
        "--" + cfg.DATA_DIR_ARG,
        metavar="Path to settings and output directory (optional)",
        type=str,
        nargs="?",
        default=Path.cwd(),
        help='path to a directory containing the "volseg-settings", data will be also be output to this location',
    )
    return parser
