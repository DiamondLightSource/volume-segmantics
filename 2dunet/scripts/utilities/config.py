"""Data to be shared across files.
"""
# Parser strings
TRAIN_DATA_ARG = "data"
LABEL_DATA_ARG = "labels"
MODEL_PTH_ARG = "model"
PREDICT_DATA_ARG = "data"
DATA_DIR_ARG = "data_dir"
# File extensions
TIFF_SUFFIXES = {".tiff", ".tif"}
HDF5_SUFFIXES = {".h5", ".hdf5", ".nxs"}
TRAIN_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
LABEL_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
MODEL_DATA_EXT = {".pytorch", ".pth"}
PREDICT_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
# TODO Required settings - check required keys are in settings files
# Logging format
LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"
# Settings yaml file locations
SETTINGS_DIR = "unet-settings"
TRAIN_SETTINGS_FN = "2d_unet_train_settings.yaml"
PREDICTION_SETTINGS_FN = "2d_unet_predict_settings.yaml"

BIG_CUDA_SIZE = 8 # GPU Memory (GB), above this value batch size is increased
BIG_CUDA_BATCH = 16 # Size of batch on big GPU
SMALL_CUDA_BATCH = 4 # Size of batch on small GPU
NUM_WORKERS = 8 # Number of parallel workers for training/validation dataloaders
PIN_CUDA_MEMORY = True # Whether to pin CUDA memory for faster data transfer 

DEFAULT_MIN_LR = 0.00075 # Learning rate to return if LR finder fails

IMAGENET_MEAN = 0.449 # Mean value for single channel imagnet normalisation
IMAGENET_STD = 0.226 # Standard deviation for single channel imagenet normalisation
 