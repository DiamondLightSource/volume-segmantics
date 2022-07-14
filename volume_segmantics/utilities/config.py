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
SETTINGS_DIR = "volseg-settings"
TRAIN_SETTINGS_FN = "2d_model_train_settings.yaml"
PREDICTION_SETTINGS_FN = "2d_model_predict_settings.yaml"

TQDM_BAR_FORMAT = "{l_bar}{bar: 30}{r_bar}{bar: -30b}"  # tqdm progress bar format

HDF5_COMPRESSION = "gzip"

BIG_CUDA_THRESHOLD = 8 # GPU Memory (GB), above this value batch size is increased
BIG_CUDA_TRAIN_BATCH = 12 # Size of training batch on big GPU
BIG_CUDA_PRED_BATCH = 4 # Size of prediction batch on big GPU
SMALL_CUDA_BATCH = 2 # Size of batch on small GPU
NUM_WORKERS = 4 # Number of parallel workers for training/validation dataloaders
PIN_CUDA_MEMORY = True # Whether to pin CUDA memory for faster data transfer
IM_SIZE_DIVISOR = 32 # Image dimensions need to be a multiple of this value
MODEL_INPUT_CHANNELS = 1 # Use 1 for grayscale input images 

DEFAULT_MIN_LR = 0.00075 # Learning rate to return if LR finder fails
LR_DIVISOR = 3 # Divide the automatically calculated learning rate (min gradient) by this magic number

IMAGENET_MEAN = 0.449 # Mean value for single channel imagnet normalisation
IMAGENET_STD = 0.226 # Standard deviation for single channel imagenet normalisation
 