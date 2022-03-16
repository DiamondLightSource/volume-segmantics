"""Data to be shared across files.
"""
# Parser strings
TRAIN_DATA_ARG = "data"
LABEL_DATA_ARG = "labels"
MODEL_PTH_ARG = "model"
PREDICT_DATA_ARG = "data"
# File extensions
TIFF_SUFFIXES = {".tiff", ".tif"}
HDF5_SUFFIXES = {".h5", ".hdf5", ".nxs"}
TRAIN_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
LABEL_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
MODEL_DATA_EXT = {".zip"}
PREDICT_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
# TODO Required settings - check required keys are in settings files
# Logging format
LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"
# Settings yaml file locations
SETTINGS_DIR = "unet-settings"
TRAIN_SETTINGS_FN = "2d_unet_train_settings.yaml"
PREDICTION_SETTINGS_FN = "2d_unet_predict_settings.yaml"


