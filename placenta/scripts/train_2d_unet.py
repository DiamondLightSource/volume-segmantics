import os
import warnings
import yaml

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


real_path = os.path.realpath(__file__ ) 
dir_path = os.path.dirname(real_path) # Extract the directory of the script
settings_path = Path(dir_path/'settings'/'2d_unet_train_settings.yaml')
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
OUT_ROOT_DIR = Path(settings_dict['out_root_dir'])-
DATA_IM_OUT_DIR = OUT_ROOT_DIR/settings_dict['data_im_dirname']
SEG_IM_OUT_DIR = OUT_ROOT_DIR/settings_dict['seg_im_out_dir']
MODEL_OUTPUT_FN = settings_dict['model_output_fn']
CODES = settings_dict['codes']
WEIGHT_DECAY = settings_dict['weight_decay'] # weight decay 
PCT_LR_INC = settings_dict['pact_lr_inc'] # the percentage of overall iterations where the LR is increasing
NUM_CYC_FROZEN = settings_dict['num_cyc_frozen'] # Number of times to run fit_one_cycle on frozen unet model
NUM_CYC_UNFROZEN = settings_dict['num_cyc_unfrozen'] # Number of times to run fit_one_cycle on unfrozen unet model

############# Data preparation #################

def da_from_data(path):    
    f = h5.File(path, 'r')
    d = f['/data']
    return da.from_array(d, chunks='auto')

def output_im(data, path, offset, crop_val, normalize=False, label=False):
    #data = data.compute()
    if normalize:
        data_st_dev = np.std(data)
        data = np.clip(data, None, data_st_dev * 3) # 99.7% of values withing 3 stdevs
        data = exposure.rescale_intensity(data, out_range='float')
        data = img_as_ubyte(data)
    if label:
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
        for val in range(shape_tup[0]):
            out_path = output_path/f"{name_prefix}_z_stack_{val}"
            output_im(data_arr[val, :, :], out_path, offset, crop_val, normalize, label)
    if axis in ['x', 'all']:
        print('Outputting x stack')
        for val in range(shape_tup[1]):
            out_path = output_path/f"{name_prefix}_x_stack_{val}"
            output_im(data_arr[:, val, :], out_path, offset, crop_val, normalize, label)
    if axis in ['y', 'all']:
        print('Outputting y stack')
        for val in range(shape_tup[2]):
            out_path = output_path/f"{name_prefix}_y_stack_{val}"
            output_im(data_arr[:, :, val], out_path, offset, crop_val, normalize, label)
    if axis not in ['x', 'y', 'z', 'all']:
        print("Axis should be one of: [all, x, y, or z]!")

# Read in the data and ground truth volumes
data_vol = da_from_data(DATA_VOL_PATH)
seg_vol = da_from_data(SEG_VOL_PATH)

# Output the normalised data slices
output_slices_to_disk('all', data_vol, DATA_IM_OUT_DIR, 'data', None, None, normalize=True)
# Output the seg slices
output_slices_to_disk('all', seg_vol, SEG_IM_OUT_DIR, 'seg', None, None, label=True)

############# Unet Training #################
# Need to define a binary label list class
class BinaryLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn)

class BinaryItemList(SegmentationItemList):
    _label_cls = BinaryLabelList

def bce_loss(logits, labels):
    logits=logits[:,1,:,:].float()
    labels = labels.squeeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    """Function taken from https://forums.fast.ai/t/automated-learning-rate-suggester/44199
        Parameters:

        lr_diff provides the interval distance by units of the “index of LR” (log transform of LRs) between the right and left bound
        loss_threshold is the maximum difference between the left and right bound’s loss values to stop the shift
        adjust_value is a coefficient to the final learning rate for pure manual adjustment
        plot is a boolean to show two plots, the LR finder’s gradient and LR finder plots as shown below
    """
    #Run the Learning Rate Finder
    model.lr_find()
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    #Search for index in gradients where loss is lowest before the loss spike
    #Initialize right and left idx using the lr_diff as a spacing unit
    #Set the local min lr as -1 to signify if threshold is too low
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value
    
    if plot:
        # plots the gradients of the losses in respect to the learning rate change
        plt.plot(loss_grad)
        plt.plot(len(losses)+l_idx, loss_grad[l_idx],markersize=10,marker='o',color='red')
        plt.ylabel("Loss")
        plt.xlabel("Index of LRs")
        plt.show()

        plt.plot(np.log10(lrs), losses)
        plt.ylabel("Loss")
        plt.xlabel("Log 10 Transform of Learning Rate")
        loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
        plt.plot(np.log10(lr_to_use), loss_coord, markersize=10,marker='o',color='red')
        plt.show()
        
    return lr_to_use

# Create a metric for asessing performance
metrics=[partial(dice, iou=True)]

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
print(f"using bs={bs}, have {free}MB of GPU RAM free")

# Create the training and test set
np.random.seed(42)
src = (BinaryItemList.from_folder(DATA_IM_OUT_DIR)
       .split_by_rand_pct()
       .label_from_func(get_y_fn, classes=CODES))
data = (src.transform(get_transforms(), size=256, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

# Create the Unet
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=WEIGHT_DECAY, loss_func=bce_loss,
callback_fns=[partial(CSVLogger, filename='unet_training_history'),
partial(SaveModelCallback, monitor='dice', mode='max', name="best_unet_model")])
#  Find a decnt learning rate and Do some learning on the frozen model
if NUM_CYC_FROZEN > 0: 
    lr_to_use = find_appropriate_lr(learn)
    learn.fit_one_cycle(NUM_CYC_FROZEN, slice(lr_to_use), pct_start=PCT_LR_INC)
if NUM_CYC_UNFROZEN > 0:
    learn.unfreeze()
    lr_to_use = find_appropriate_lr(learn)
    learn.fit_one_cycle(NUM_CYC_FROZEN, slice(lr_to_use), pct_start=PCT_LR_INC)

# Save the model
model_out = OUT_ROOT_DIR/MODEL_OUTPUT_FN
print(f"Exporting trained model to {model_out}")
learn.export(model_out)
