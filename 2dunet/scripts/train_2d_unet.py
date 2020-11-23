import os
import sys
import warnings
import yaml
import glob
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


real_path = os.path.realpath(__file__) 
dir_path = os.path.dirname(real_path)  # Extract the directory of the script
settings_path = Path(dir_path, 'settings', '2d_unet_train_settings.yaml')
print(f"Loading settings from {settings_path}")
if settings_path.exists():
    with open(settings_path, 'r') as stream:
        settings_dict = yaml.safe_load(stream)
else:
    print("Couldn't find settings file... Exiting!")
    sys.exit(1)

# Input/output Paths
IN_ROOT_DIR = Path(settings_dict['in_root_dir'])
DATA_VOL_PATH = IN_ROOT_DIR/settings_dict['data_vol_filename']
SEG_VOL_PATH = IN_ROOT_DIR/settings_dict['seg_vol_filename']
OUT_ROOT_DIR = Path(settings_dict['out_root_dir'])
DATA_IM_OUT_DIR = OUT_ROOT_DIR/settings_dict['data_im_dirname']
SEG_IM_OUT_DIR = OUT_ROOT_DIR/settings_dict['seg_im_out_dirname']
MODEL_OUTPUT_FN = settings_dict['model_output_fn']
CODES = settings_dict['codes']
NORMALIZE = settings_dict['normalize']
IMAGE_SIZE = settings_dict['image_size']  # Image size used in the Unet
WEIGHT_DECAY = float(settings_dict['weight_decay'])  # weight decay 
PCT_LR_INC = settings_dict['pct_lr_inc']  # the percentage of overall iterations where the LR is increasin
NUM_CYC_FROZEN = settings_dict['num_cyc_frozen']  # Number of times to run fit_one_cycle on frozen unet model
NUM_CYC_UNFROZEN = settings_dict['num_cyc_unfrozen']  # Number of times to run fit_one_cycle on unfrozen unet model
multilabel = False   # Set flag to false for binary segmentation true for n classes > 2
############# Data preparation #################

def da_from_data(path):    
    f = h5.File(path, 'r')
    d = f['/data']
    return da.from_array(d, chunks='auto')

def output_im(data, path, offset, crop_val, normalize=False, label=False):
    if isinstance(data, da.core.Array):
        data = data.compute()
    if normalize:
        data_st_dev = np.std(data)
        data = np.clip(data, None, data_st_dev * 3) # 99.7% of values withing 3 stdevs
        data = exposure.rescale_intensity(data)
        data = img_as_ubyte(data)
    if label and not multilabel:
        data[data > 1] = 1
    # Crop the image
    if offset and crop_val:
        data = data[offset:crop_val, offset:crop_val]
    io.imsave(f'{path}.png', data)

def output_slices_to_disk(axis, data_path, output_path, name_prefix, offset, crop_val, normalize=False, label=False):
    data_arr = data_path
    shape_tup = data_arr.shape
    # There has to be a cleverer way to do this!
    if axis in ['z', 'all']:
        print('Outputting z stack')
        for val in tqdm(range(shape_tup[0])):
            out_path = output_path/f"{name_prefix}_z_stack_{val}"
            output_im(data_arr[val, :, :], out_path, offset, crop_val, normalize, label)
    if axis in ['x', 'all']:
        print('Outputting x stack')
        for val in tqdm(range(shape_tup[1])):
            out_path = output_path/f"{name_prefix}_x_stack_{val}"
            output_im(data_arr[:, val, :], out_path, offset, crop_val, normalize, label)
    if axis in ['y', 'all']:
        print('Outputting y stack')
        for val in tqdm(range(shape_tup[2])):
            out_path = output_path/f"{name_prefix}_y_stack_{val}"
            output_im(data_arr[:, :, val], out_path, offset, crop_val, normalize, label)
    if axis not in ['x', 'y', 'z', 'all']:
        print("Axis should be one of: [all, x, y, or z]!")

# Read in the data and ground truth volumes
data_vol = da_from_data(DATA_VOL_PATH)
seg_vol = da_from_data(SEG_VOL_PATH)

# Check that we have the right number of classes in the seg volume
seg_classes = np.unique(seg_vol.compute())
num_seg_classes = len(seg_classes)
print(f"Number of classes in segmentation dataset: {num_seg_classes}")
num_codes = len(CODES)
if num_seg_classes != num_codes:
    print(f"{num_codes} classes were specified in the settings file, however "
          f"the data contains {num_seg_classes} classes. Exiting.")
    sys.exit(1)
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

# Output the normalised data slices
print("Slicing data volume in 3 directions and outputting to disk")
os.makedirs(DATA_IM_OUT_DIR, exist_ok=True)
output_slices_to_disk('all', data_vol, DATA_IM_OUT_DIR, 'data', None, None,
                      normalize=NORMALIZE)
# Output the seg slices
print("Slicing segemented volume in 3 directions and outputting to disk")
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
    print(f"Performing multilabel segmentation since there are "
          f"{num_seg_classes} classes")
    metrics = accuracy
    monitor = 'accuracy'
    loss_func = None
else:
    print(f"Performing binary segmentation since there are "
          f"{num_seg_classes} classes")
    metrics = [partial(dice, iou=True)]
    monitor = 'dice'
    loss_func = bce_loss
    
 # Create the training and test set
print("Creating training dataset from saved images.")
src = (SegmentationItemList.from_folder(DATA_IM_OUT_DIR)
       .split_by_rand_pct()
       .label_from_func(get_y_fn, classes=CODES))
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
model_out = OUT_ROOT_DIR/MODEL_OUTPUT_FN
print(f"Exporting trained model to {model_out}")
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
plt.suptitle(f"Predictions for {MODEL_OUTPUT_FN}", fontsize=16)
plt_out_pth = OUT_ROOT_DIR/f'{model_out.stem}_prediction_image.png' 
print(f"Saving example image predictions to {plt_out_pth}")  
plt.savefig(plt_out_pth, dpi=300)

data_ims = glob.glob(f"{str(DATA_IM_OUT_DIR) + '/*.png'}")
print(f"Deleting {len(data_ims)} image slices")
for fn in data_ims:
    os.remove(fn)

seg_ims = glob.glob(f"{str(SEG_IM_OUT_DIR) + '/*.png'}")
print(f"Deleting {len(seg_ims)} ground truth slices")
for fn in seg_ims:
    os.remove(fn)
