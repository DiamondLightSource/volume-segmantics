import argparse
import os
import sys
import warnings
import yaml
import glob
from datetime import date
from tqdm import tqdm

import dask.array as da
import h5py as h5
import numpy as np
import torch.nn.functional as F
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.vision import *
from skimage import exposure, img_as_float, img_as_ubyte, io
from skimage.transform import resize

warnings.filterwarnings("ignore", category=UserWarning)

root_path = Path.cwd()  # For module load script, use the CWD
settings_path = Path(root_path, 'unet-settings', '2d_unet_train_settings.yaml')
print(f"Loading settings from {settings_path}")
if settings_path.exists():
    with open(settings_path, 'r') as stream:
        settings_dict = yaml.safe_load(stream)
else:
    print("Couldn't find settings file... Exiting!")
    sys.exit(1)

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/data/file.h5] [path/to/segmentation/file.h5]",
        description="Train a 2d U-net model on the 3d data and corresponding"\
        "segmentation provided in the files."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('data_vol_path', metavar='Image data file path', type=str,
                        help='the path to an HDF5 file containing the imaging data volume.')
    parser.add_argument('seg_vol_path', metavar='Segmentation file path', type=str,
                        help='the path to an HDF5 file containing a segmented volume.')
    return parser

parser = init_argparse()
args = parser.parse_args()
data_vol_path = Path(args.data_vol_path)
seg_vol_path = Path(args.seg_vol_path)
if not seg_vol_path.is_file() or not data_vol_path.is_file():
    print(f"One or more of the given paths does not appear to specify a file. Exiting!")
    sys.exit(1)

# Input/output Paths
DATA_IM_OUT_DIR = root_path/settings_dict['data_im_dirname']
SEG_IM_OUT_DIR = root_path/settings_dict['seg_im_out_dirname']
MODEL_OUTPUT_FN = settings_dict['model_output_fn']
NORMALISE = settings_dict['normalise']
IMAGE_SIZE = settings_dict['image_size']  # Image size used in the Unet
WEIGHT_DECAY = float(settings_dict['weight_decay'])  # weight decay 
PCT_LR_INC = settings_dict['pct_lr_inc']  # the percentage of overall iterations where the LR is increasin
NUM_CYC_FROZEN = settings_dict['num_cyc_frozen']  # Number of times to run fit_one_cycle on frozen unet model
NUM_CYC_UNFROZEN = settings_dict['num_cyc_unfrozen']  # Number of times to run fit_one_cycle on unfrozen unet model
ST_DEV_FACTOR = 2.575 # 99% of values lie within 2.575 stdevs of the mean
multilabel = False   # Set flag to false for binary segmentation true for n classes > 2
############# Data preparation #################

def da_from_data(path):    
    f = h5.File(path, 'r')
    d = f['/data']
    return da.from_array(d, chunks='auto')

def output_im(data, path, offset, crop_val, label=False):
    if isinstance(data, da.core.Array):
        data = data.compute()
    if label and not multilabel:
        data[data > 1] = 1
    # Crop the image
    if offset and crop_val:
        data = data[offset:crop_val, offset:crop_val]
    io.imsave(f'{path}.png', data)

def output_slices_to_disk(axis, data_path, output_path, name_prefix, offset, crop_val, label=False):
    data_arr = data_path
    shape_tup = data_arr.shape
    # There has to be a cleverer way to do this!
    if axis in ['z', 'all']:
        print('Outputting z stack')
        for val in tqdm(range(shape_tup[0])):
            out_path = output_path/f"{name_prefix}_z_stack_{val}"
            output_im(data_arr[val, :, :], out_path, offset, crop_val, label)
    if axis in ['x', 'all']:
        print('Outputting x stack')
        for val in tqdm(range(shape_tup[1])):
            out_path = output_path/f"{name_prefix}_x_stack_{val}"
            output_im(data_arr[:, val, :], out_path, offset, crop_val, label)
    if axis in ['y', 'all']:
        print('Outputting y stack')
        for val in tqdm(range(shape_tup[2])):
            out_path = output_path/f"{name_prefix}_y_stack_{val}"
            output_im(data_arr[:, :, val], out_path, offset, crop_val, label)
    if axis not in ['x', 'y', 'z', 'all']:
        print("Axis should be one of: [all, x, y, or z]!")

# Read in the data and ground truth volumes
data_vol = da_from_data(data_vol_path)
seg_vol = da_from_data(seg_vol_path)

seg_classes = np.unique(seg_vol.compute())
num_seg_classes = len(seg_classes)
print(f"Number of classes in segmentation dataset: {num_seg_classes}")
print("These classes are:", *seg_classes, sep='\n')
if num_seg_classes > 2:
    multilabel = True
    
# Make sure label classes start from 0
def fix_label_classes(seg_vol, seg_classes):
    if isinstance(seg_vol, da.core.Array):
        seg_vol = seg_vol.compute()
    for idx, current in enumerate(seg_classes):
        seg_vol[seg_vol == current] = idx
    return seg_vol

if seg_classes[0] != 0:
    print("Fixing label classes")
    seg_vol = fix_label_classes(seg_vol, seg_classes)

codes = [f"label_val{i}" for i in seg_classes]


def clip_to_uint8(data, st_dev_factor):
    data_st_dev = np.std(data)
    data_mean = np.mean(data)
    num_vox = np.prod(data.shape)
    lower_bound = data_mean - (data_st_dev * st_dev_factor)
    upper_bound = data_mean + (data_st_dev * st_dev_factor)
    gt_ub = (data > upper_bound).sum()
    lt_lb =(data < lower_bound).sum()
    print(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
    print(
        f"Number of voxels above upper bound to be clipped {gt_ub} - percentage {gt_ub/num_vox * 100:.3f}%")
    print(
        f"Number of voxels below lower bound to be clipped {lt_lb} - percentage {lt_lb/num_vox * 100:.3f}%")
    data = np.clip(data, lower_bound, upper_bound)
    data = exposure.rescale_intensity(data, out_range='float')
    return img_as_ubyte(data)

# Output the data slices
if NORMALISE:
    print('Normalising the image data volume and downsampling to uint8.')
    data_vol = clip_to_uint8(data_vol.compute(), ST_DEV_FACTOR)
print("Slicing data volume in 3 directions and saving slices to disk.")
os.makedirs(DATA_IM_OUT_DIR, exist_ok=True)
output_slices_to_disk('all', data_vol, DATA_IM_OUT_DIR, 'data', None, None)
# Output the seg slices
print("Slicing segmented volume in 3 directions and saving slices to disk.")
os.makedirs(SEG_IM_OUT_DIR, exist_ok=True)
output_slices_to_disk('all', seg_vol, SEG_IM_OUT_DIR, 'seg', None, None, label=True)

############# Unet Training #################
# Need to define a binary label list class for binary segmenation
class BinaryLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn)

class BinaryItemList(SegmentationItemList):
    _label_cls = BinaryLabelList

def bce_loss(logits, labels):
    logits=logits[:,1,:,:].float()
    labels = labels.squeeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1) -> float:
    """Function taken from https://forums.fast.ai/t/automated-learning-rate-suggester/44199
        Parameters:

        lr_diff provides the interval distance by units of the “index of LR” (log transform of LRs) between the right and left bound
        loss_threshold is the maximum difference between the left and right bound’s loss values to stop the shift
        adjust_value is a coefficient to the final learning rate for pure manual adjustment
        plot is a boolean to show two plots, the LR finder’s gradient and LR finder plots as shown below
    """
    #Run the Learning Rate Finder
    lr_find(model)
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    #Search for index in gradients where loss is lowest before the loss spike
    #Initialize right and left idx using the lr_diff as a spacing unit
    #Set the local min lr as -1 to signify if threshold is too low
    local_min_lr = 0.001 # Add as default value to fix bug
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value
        
    return lr_to_use

# Create a metric for asessing performance
def accuracy(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1) == target).float().mean()


# Get the data filenames
fnames = get_image_files(DATA_IM_OUT_DIR)
# Get the seg filenames
lbl_names = get_image_files(SEG_IM_OUT_DIR)
# function to convert between data and seg filenames
get_y_fn = lambda x: SEG_IM_OUT_DIR/f'{"seg" + x.stem[4:]}{x.suffix}'
# Choose a batchsize
free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free} MB of GPU RAM free")

if multilabel:
    print(f"Training for multilabel segmentation since there are "
          f"{num_seg_classes} classes")
    metrics = accuracy
    monitor = 'accuracy'
    loss_func = None
else:
    print(f"Training for binary segmentation since there are "
          f"{num_seg_classes} classes")
    metrics = [partial(dice, iou=True)]
    monitor = 'dice'
    loss_func = bce_loss
    
 # Create the training and test set
print("Creating training dataset from saved images.")
src = (SegmentationItemList.from_folder(DATA_IM_OUT_DIR)
       .split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=IMAGE_SIZE, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
# Create the Unet
print("Creating Unet for training.")
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=WEIGHT_DECAY,
                     loss_func=loss_func,
                     callback_fns=[partial(CSVLogger,
                                           filename='unet_training_history',
                                           append=True),
                                   partial(SaveModelCallback,
                                           monitor=monitor, mode='max',
                                           name="best_unet_model")])

#  Find a decent learning rate and Do some learning on the frozen model
if NUM_CYC_FROZEN > 0: 
    print("Finding learning rate for frozen Unet model.")
    lr_to_use = find_appropriate_lr(learn)
    print(f"Training frozen Unet for {NUM_CYC_FROZEN} cycles with learning rate of {lr_to_use}.")
    learn.fit_one_cycle(NUM_CYC_FROZEN, slice(lr_to_use/50, lr_to_use), pct_start=PCT_LR_INC)
if NUM_CYC_UNFROZEN > 0:
    learn.unfreeze()
    print("Finding learning rate for unfrozen Unet model.")
    lr_to_use = find_appropriate_lr(learn)
    print(f"Training unfrozen Unet for {NUM_CYC_UNFROZEN} cycles with learning rate of {lr_to_use}.")
    learn.fit_one_cycle(NUM_CYC_UNFROZEN, slice(lr_to_use/50, lr_to_use), pct_start=PCT_LR_INC)

# Save the model
model_fn = f"{date.today()}_{MODEL_OUTPUT_FN}"
model_out = Path(root_path, model_fn)
learn.export(model_out)

# Output a figure showing predictions from the validation dataset
learn.data.single_ds.tfmargs['size'] = None # Remove the restriction on the model prediction size
filename_list = data.valid_ds.items[:3]
img_list = []
pred_list = []
gt_list = []
for fn in filename_list:
    img_list.append(open_image(fn))
    gt_list.append(io.imread(get_y_fn(fn)))
for img in img_list:
    pred_list.append(img_as_ubyte(learn.predict(img)[1][0]))
# Horrible conversion from Fastai image to unit8 data array
img_list = [img_as_ubyte(exposure.rescale_intensity(x.data.numpy()[0, :, :])) for x in img_list]

# Create the plot
fig=plt.figure(figsize=(12, 12))
columns = 3
rows = 3
j= 0
for i in range(columns*rows)[::3]:
    img = img_list[j]
    gt = gt_list[j]
    pred = pred_list[j]
    col1 = fig.add_subplot(rows, columns, i + 1)
    plt.imshow(img, cmap='gray')
    col2 = fig.add_subplot(rows, columns, i + 2)
    plt.imshow(gt, cmap='gray')
    col3 = fig.add_subplot(rows, columns, i + 3)
    plt.imshow(pred, cmap='gray')
    j += 1
    if i == 0:
        col1.title.set_text('Data')
        col2.title.set_text('Ground Truth')
        col3.title.set_text('Prediction')
plt.suptitle(f"Predictions for {model_fn}", fontsize=16)
plt_out_pth = root_path/f'{model_out.stem}_prediction_image.png' 
print(f"Saving example image predictions to {plt_out_pth}")  
plt.savefig(plt_out_pth, dpi=300)

data_ims = glob.glob(f"{str(DATA_IM_OUT_DIR) + '/*.png'}")
print(f"Deleting {len(data_ims)} image slices")
for fn in data_ims:
    os.remove(fn)

seg_ims = glob.glob(f"{str(SEG_IM_OUT_DIR) + '/*.png'}")
print(f"Deleting {len(seg_ims)} segmentation slices")
for fn in seg_ims:
    os.remove(fn)
print(f"Deleting the empty segmentation image directory")
os.rmdir(SEG_IM_OUT_DIR)
