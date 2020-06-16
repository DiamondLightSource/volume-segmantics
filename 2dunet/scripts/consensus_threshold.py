"""
    Takes an imput data volume and a 2D Unet trained for binary segmentation
    Slices the data volume in the three orthogonal planes and predicts output for each slice
    The predictions are recombined into 3D volumes and then summed
    The input data volume is rotated by 90 degrees before the slicing and prediction steps are performed again
    This is repeated until 4 rotations have been been performed
    All the volumes are summed to give a prediction that is the sum of predictions in 12 different directions,
    A list of threshold values for a consensus cutoff is used to give a number of output volumes

"""

import os
import sys
import warnings
from datetime import date
from pathlib import Path

import dask.array as da
import h5py as h5
import numpy as np
import yaml

warnings.filterwarnings("ignore", category=UserWarning)


real_path = os.path.realpath(__file__ ) 
dir_path = os.path.dirname(real_path) # Extract the directory of the script
settings_path = Path(dir_path,'settings','2d_unet_threshold_settings.yaml')
print(f"Loading settings from {settings_path}")
if settings_path.exists():
    with open(settings_path, 'r') as stream:
        settings_dict = yaml.safe_load(stream)
else:
    print("Couldn't find settings file... Exiting!")
    sys.exit(1)


"""
    PREDICT_DIR - Filename of the HDF5 volume to be thresholded - should be the 4_volumes_combined file
    DATA_VOL_FN - Filename of the HDF5 volume to be thresholded - should be the 4_volumes_combined file
    CONSENSUS_VALS - List of consensus cutoff values for agreement between volumes
    e.g. if 10 is in the list a volume will be output thresholded on consensus between 10 volumes

"""


PREDICT_DIR = Path(settings_dict['predict_folder_path'])
DATA_VOL_FN = Path(settings_dict['volume_name'])
CONSENSUS_VALS = map(int, settings_dict['consensus_vals'])



######## Utility functions ############
def da_from_data(path):    
    f = h5.File(path, 'r')
    d = f['/data']
    return da.from_array(d, chunks='auto')

def threshold(input_path, range_list):
    for val in range_list:
        combined = da_from_data(input_path)
        combined_out = input_path.parent/f'{date.today()}_combined_thresh_cutoff_{val}.h5'
        combined[combined < val] = 0
        combined[combined >= val] = 255
        print(f'Writing to {combined_out}')
        combined.to_hdf5(combined_out, '/data')


######## Do stuff here #########
threshold(PREDICT_DIR/DATA_VOL_FN, CONSENSUS_VALS)
