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
from datetime import date
import re
import yaml
from tqdm import tqdm
import numpy as np
import dask.array as da
import h5py as h5
from fastai.vision import *
from fastai.utils.mem import *
from skimage import img_as_ubyte, io, exposure, img_as_float
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


real_path = os.path.realpath(__file__ ) 
dir_path = os.path.dirname(real_path) # Extract the directory of the script
settings_path = Path(dir_path,'settings','2d_unet_predict_settings.yaml')
print(f"Loading settings from {settings_path}")
if settings_path.exists():
    with open(settings_path, 'r') as stream:
        settings_dict = yaml.safe_load(stream)
else:
    print("Couldn't find settings file... Exiting!")
    sys.exit(1)


"""
    ROOT_PATH - Root filepath for output directories, folder will be created
    DATA_VOL_PATH - Path to the HDF5 volume to be segmented. Data should be in /data inside the file
    LEARNER_ROOT_PATH - Path to the folder containing the model file
    LEARNER_FILE - Filename of the pickled 2d Unet model file. Needs to have been trained using BCE loss. For binary segmentation only
    CONSENSUS_VALS - List of consensus cutoff values for agreement between volumes
    e.g. if 10 is in the list a volume will be output thresholded on consensus between 10 volumes

"""


ROOT_PATH = Path(settings_dict['root_path'])
DATA_VOL_PATH = Path(settings_dict['data_vol_path'])
LEARNER_ROOT_PATH = Path(settings_dict['model_root_path'])
LEARNER_FILE = settings_dict['model_file']
CONSENSUS_VALS = map(int, settings_dict['consensus_vals'])
NORMALISE = settings_dict['normalise']


######## Utility functions ############
makedirs = partial(os.makedirs, exist_ok=True)
def da_from_data(path):    
    f = h5.File(path, 'r')
    d = f['/data']
    return da.from_array(d, chunks='auto')

# Needed because prediction doesn't work on odd sized images
def fix_odd_sides(example_image):
    if (list(example_image.size)[0] % 2) != 0:
        example_image = crop_pad(example_image, 
                            size=(list(example_image.size)[0]+1, list(example_image.size)[1]),
                            padding_mode = 'reflection')

    if (list(example_image.size)[1] % 2) != 0:
        example_image = crop_pad(example_image, 
                            size=(list(example_image.size)[0], list(example_image.size)[1] + 1),
                            padding_mode = 'reflection')

def predict_single_slice(learn, axis, val, data, output_path, normalise=False):
    #data = data.compute()
    data = img_as_float(data)
    if normalise:
        data_st_dev = np.std(data)
        data = np.clip(data, None, data_st_dev * 3) # 99.7% of values withing 3 stdevs
        data = exposure.rescale_intensity(data, out_range='float')
    img = Image(pil2tensor(data, dtype=np.float32))
    fix_odd_sides(img)
    prediction = learn.predict(img)
    pred_slice = img_as_ubyte(prediction[1][0])
    io.imsave(output_path/f"unet_prediction_{axis}_stack_{val}.png", pred_slice)

def predict_orthog_slices_to_disk(learn, axis, data_arr, output_path):
    """Outputs slices from data or ground truth seg volumes sliced in any or all three of the orthogonal planes"""
    data_shape = data_arr.shape
    # There has to be a cleverer way to do this!
    if axis in ['z', 'all']:
        for val in tqdm(range(data_shape[0]), desc='Predicting z stack', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            predict_single_slice(learn, 'z', val, data_arr[val, :, :], output_path, NORMALISE)
    if axis in ['x', 'all']:
        for val in tqdm(range(data_shape[1]), desc='Predicting x stack', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            predict_single_slice(learn, 'x', val, data_arr[:, val, :], output_path, NORMALISE)                    
    if axis in ['y', 'all']:
        for val in tqdm(range(data_shape[2]), desc='Predicting y stack', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            predict_single_slice(learn, 'y', val, data_arr[:, :, val], output_path, NORMALISE)
    if axis not in ['x', 'y', 'z', 'all']:
        print("Axis should be one of: [all, x, y, or z]!")

def setup_folder_stucture(root_path):  
    non_rotated = root_path/f'{date.today()}_non_rotated_seg_slices'
    rot_90_seg = root_path/f'{date.today()}_rot_90_seg_slices'
    rot_180_seg = root_path/f'{date.today()}_rot_180_seg_slices'
    rot_270_seg = root_path/f'{date.today()}_rot_270_seg_slices'
    
    dir_list = [
        ('non_rotated', non_rotated),
        ('rot_90_seg', rot_90_seg),
        ('rot_180_seg', rot_180_seg),
        ('rot_270_seg', rot_270_seg)
    ]
    for key, dir_path in dir_list:
        makedirs(dir_path)
    return dir_list

# Need the loss in order to load the learner..
def bce_loss(logits, labels):
    logits=logits[:,1,:,:].float()
    labels = labels.squeeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels)

class BinaryLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn)

class BinaryItemList(SegmentationItemList):
    _label_cls = BinaryLabelList

def combine_slices_to_vol(folder_path):
    output_path_list = []
    file_list = folder_path.ls()
    axis_list = ['z', 'x', 'y']
    number_regex = re.compile(r'\_(\d+)\.png')
    for axis in axis_list:
        axis_files = [x for x in file_list if re.search(f'\_({axis})\_', str(x))]
        print(f"Creating volume from {axis} stack")
        print(f'{len(axis_files)} files found')
        first_im = open_image(axis_files[0])
        shape_tuple = first_im.shape
        z_dim = len(axis_files)
        x_dim = shape_tuple[1]
        y_dim = shape_tuple[2]
        data_vol = np.empty([z_dim, x_dim, y_dim], dtype=np.uint8)
        for filename in axis_files:
            m = number_regex.search(str(filename))
            pos = int(m.group(1))
            im_data = io.imread(filename)
            data_vol[pos, :, :] = im_data
        if axis == 'x':
            data_vol = np.swapaxes(data_vol, 0, 1)
        if axis == 'y':
            data_vol = np.swapaxes(data_vol, 0, 2)
            data_vol = np.swapaxes(data_vol, 0, 1)
        output_path = folder_path/f'{axis}_axis_seg_combined.h5'
        output_path_list.append(output_path)
        print(f'Outputting volume to {output_path}')
        with h5.File(output_path, 'w') as f:
            f['/data'] = data_vol
        # Delete the images
        print(f"Deleting {len(axis_files)} image files for axis {axis}")
        for filename in axis_files:
            os.remove(filename)
    return output_path_list

def combine_vols(output_path_list, k, prefix, final=False):
    num_vols = len(output_path_list)
    combined = da_from_data(output_path_list[0])
    for subsequent in output_path_list[1:]:
        combined += da_from_data(subsequent)
    combined_out_path = output_path_list[0].parent.parent/f'{date.today()}_{prefix}_{num_vols}_volumes_combined.h5'
    if final:
        combined_out_path = output_path_list[0].parent/f'{date.today()}_{prefix}_{num_vols}_volumes_combined.h5'
    print(f'Outputting the {num_vols} combined volumes to {combined_out_path}')
    combined = combined.compute()
    combined = np.rot90(combined, k, (1, 0))
    with h5.File(combined_out_path, 'w') as f:
        f['/data'] = combined
    return combined_out_path

def threshold(input_path, range_list):
    for val in range_list:
        combined = da_from_data(input_path)
        combined_out = input_path.parent/f'{date.today()}_combined_thresh_cutoff_{val}.h5'
        combined[combined < val] = 0
        combined[combined >= val] = 255
        print(f'Writing to {combined_out}')
        combined.to_hdf5(combined_out, '/data')


######## Do stuff here #########

# Make a root directory for the ouitput
makedirs(ROOT_PATH)
# Load the data volume and the model
data_arr = da_from_data(DATA_VOL_PATH)
learn = load_learner(LEARNER_ROOT_PATH, LEARNER_FILE)
# Remove the restriction on the model prediction size
learn.data.single_ds.tfmargs['size'] = None
# Run the loop to do repeated prediction and recombination steps
axis = 'all'
dir_list = setup_folder_stucture(ROOT_PATH)
combined_vol_paths = []
data_arr = data_arr.compute()
for k in tqdm(range(4), ncols=100, desc='Total progress'):
    key, output_path = dir_list[k]
    print(f'Key : {key}, output : {output_path}')
    print(f'Rotating volume {k * 90} degrees')
    rotated = np.rot90(data_arr, k)
    predict_orthog_slices_to_disk(learn, axis, rotated, output_path)
    output_path_list = combine_slices_to_vol(output_path)
    fp = combine_vols(output_path_list, k, key)
    combined_vol_paths.append(fp)
# Combine all the volumes
final_combined = combine_vols(combined_vol_paths, 0, 'final', True)
threshold(final_combined, CONSENSUS_VALS)