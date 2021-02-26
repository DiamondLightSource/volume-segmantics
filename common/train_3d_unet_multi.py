"""Script to train a 3D-unet for binary segmentation when given data and ground
 truth HDF5 input volumes. A separate set of data and ground truth volumes are
 required for validation.
"""
import csv
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

import h5py as h5
import numpy as np
import termplotlib as tpl
import torch
import torchio
import yaml
from matplotlib import pyplot as plt
from pytorch3dunet.unet3d.losses import (BCEDiceLoss, DiceLoss,
                                         GeneralizedDiceLoss)
from pytorch3dunet.unet3d.metrics import GenericAveragePrecision, MeanIoU
from pytorch3dunet.unet3d.model import ResidualUNet3D
from torch import nn as nn
from torch.utils.data import DataLoader
from torchio import DATA
from torchio.data.inference import GridSampler
from torchio.transforms import (Compose, OneOf, RandomAffine, RandomBlur,
                                RandomElasticDeformation, RandomFlip,
                                RandomNoise, RescaleIntensity)
from tqdm import tqdm

from utilities.early_stopping import EarlyStopping

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)  # Extract the directory of the script
settings_path = Path(dir_path, 'settings',
                     '3d_unet_train_settings_twolabel.yaml')
print(f"Loading settings from {settings_path}")
if settings_path.exists():
    with open(settings_path, 'r') as stream:
        settings_dict = yaml.safe_load(stream)
else:
    print("Couldn't find settings file... Exiting!")
    sys.exit(1)

CUDA_DEVICE = str(settings_dict['cuda_device'])  # Select a particular GPU
STARTING_LR = float(settings_dict['starting_lr'])
END_LR = float(settings_dict['end_lr'])
LR_FIND_EPOCHS = settings_dict['lr_find_epochs']
DATA_DIR = Path(settings_dict['data_dir'])
TRAIN_DATA = DATA_DIR/settings_dict['train_data']
TRAIN_SEG = DATA_DIR/settings_dict['train_seg']
VALID_DATA = DATA_DIR/settings_dict['valid_data']
VALID_SEG = DATA_DIR/settings_dict['valid_seg']
PATCH_SIZE = tuple(settings_dict['patch_size'])
TRAIN_PATCHES = settings_dict['train_patches']
VALID_PATCHES = settings_dict['valid_patches']
MAX_QUEUE_LENGTH = settings_dict['max_queue_length']
NUM_WORKERS = settings_dict['num_workers']
MODEL_DICT = settings_dict['model']
MODEL_OUT_FN = settings_dict['model_out_fn']
NUM_EPOCHS = settings_dict['num_epochs']
PATIENCE = settings_dict['patience']
THRESH_VAL = settings_dict['thresh_val']
LOSS_CRITERION = settings_dict['loss_criterion']
EVAL_METRIC = settings_dict['eval_metric']
ALPHA = settings_dict['alpha']
BETA = settings_dict['beta']
DEVICE_NUM = 0  # Once single GPU slected, default device will always be 0
BAR_FORMAT = "{l_bar}{bar: 30}{r_bar}{bar: -30b}"  # tqdm progress bar

def create_unet_on_device(device, model_dict):
    unet = ResidualUNet3D(**model_dict)
    print(f"Sending the model to device {CUDA_DEVICE}")
    return unet.to(device)


def tensor_from_hdf5(file_path, data_path):
    with h5.File(file_path, 'r') as f:
        tens = torch.from_numpy(f[data_path][()])
    return tens


def numpy_from_hdf5(path, hdf5_path='/data', nexus=False, lazy=False):
    """Returns a numpy array when given a path to an HDF5 file.

    The data is assumed to be found in '/data' in the file.

    Args:
        path(pathlib.Path): The path to the HDF5 file.
        hdf5_path (str): The internal HDF5 path to the data.

    Returns:
        numpy.array: A numpy array object for the data stored in the HDF5 file.
    """

    f = h5.File(path, 'r')
    if nexus:
        try:
            data = f['processed/result/data']
        except KeyError:
            print(
                "NXS file: Couldn't find data at 'processed/result/data' trying another path.")
            try:
                data = f['entry/final_result_tomo/data']
            except KeyError:
                print(
                    "NXS file: Could not find entry at entry/final_result_tomo/data, exiting!")
                sys.exit(1)
    else:
        data = f[hdf5_path]
    if not lazy:
        data = data[()]
        f.close()
        return data
    return data, f


def prepare_batch(batch, device):
    inputs = batch['data'][DATA].to(device)
    # if multilabel:
    targets = batch['label'][DATA]
    # Split the labels into channels - multilabel
    targets = targets.squeeze()
    channels = []
    for label_num in range(NUM_CLASSES):
        channel = torch.zeros_like(targets)
        channel[targets == label_num] = 1
        channels.append(channel)
    targets = torch.stack(channels, 1).to(
        device, dtype=torch.uint8)
    return inputs, targets

############################################
# Functions to find optimum learning rate ##
# Borrows from                            ##
# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
############################################


def lr_lambda(x):
    """Exponentially increase learning rate as part of strategy to find the
    optimum.
    Taken from
    https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
    """
    return math.exp(
        x * math.log(END_LR / STARTING_LR) /
        (LR_FIND_EPOCHS * len(training_loader))
    )


def lr_finder(model, training_loader, optimizer, lr_scheduler,
              smoothing=0.05, plt_fig=True):
    lr_find_loss = []
    lr_find_lr = []
    iters = 0

    model.train()
    print(f"Training for {LR_FIND_EPOCHS} epochs to create a learning "
          "rate plot.")
    for i in range(LR_FIND_EPOCHS):
        for batch in tqdm(training_loader, desc=f'Epoch {i + 1}, batch number',
                          bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
            inputs, targets = prepare_batch(batch, DEVICE_NUM)
            optimizer.zero_grad()
            output = model(inputs)
            if LOSS_CRITERION == 'CrossEntropyLoss':
                loss = loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = loss_criterion(output, targets)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)
            if iters == 0:
                lr_find_loss.append(loss)
            else:
                loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss)
            if loss > 1 and iters > len(training_loader)// 1.333:
                break
            iters += 1

    if plt_fig:
        fig = tpl.figure()
        fig.plot(np.log10(lr_find_lr), lr_find_loss, width=50,
                 height=30, xlabel='Log10 Learning Rate')
        fig.show()

    return lr_find_loss, lr_find_lr


def find_appropriate_lr(model, lr_find_loss, lr_find_lr, lr_diff=6,
                        loss_threshold=.05, adjust_value=0.75):
    """Function taken from
    https://forums.fast.ai/t/automated-learning-rate-suggester/44199
        Parameters:

        lr_diff provides the interval distance by units of the “index of LR”
        (log transform of LRs) between the right and left bound
        loss_threshold is the maximum difference between the left and right
        bound’s loss values to stop the shift
        adjust_value is a coefficient to the final learning rate for pure
        manual adjustment
    """
    # Get loss values and their corresponding gradients, and get lr values
    losses = np.array(lr_find_loss)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = lr_find_lr

    # Search for index in gradients where loss is lowest before the loss spike
    # Initialize right and left idx using the lr_diff as a spacing unit
    # Set the local min lr as -1 to signify if threshold is too low
    local_min_lr = 0.001  # Add as default value to fix bug
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx])
                                       > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value

    return lr_to_use

##################
# Model Training #
##################


def train_model(model, optimizer, lr_to_use, training_loader, valid_loader,
                loss_criterion, eval_criterion, patience, n_epochs,
                output_path):
    """Train model with early stopping once the validation loss stops improving
    as described in https://github.com/Bjarten/early-stopping-pytorch

    Args:
        model (torch.nn.Module): Unet model for training
        optimizer (torch.optim.Optimizer): Optimizer for model training
        lr_to_use (float): Maximum learning rate to use in One Cycle cyclical
        learning rate policy
        training_loader (torch.utils.data.dataloader): Dataloader for patches
        from the training volume
        valid_loader (torch.utils.data.dataloader): Dataloader for patches from
        the validation volume
        loss_criterion (torch.nn.Module): Criterion to be used for measuring
        model loss
        eval_criterion (Class): Class to calculate evaluation metric
        e.g. MeanIOU
        patience (int): Number of epochs to wait before stopping training
        after validation loss stops decreasing
        n_epochs ([type]): Maximum number of epochs to train for
        output_path (tuple ): Tuple containing Model with lowest validation
        loss, list of average training losses, list of average validation
        losses, list of average evaluation scores

    Returns:
        [type]: [description]
    """
    # to track the training loss
    train_losses = []
    # to track the validation loss
    valid_losses = []
    # to track the average training loss per epoch
    avg_train_losses = []
    # to track the average validation loss per epoch
    avg_valid_losses = []
    # to track the evaluation score
    eval_scores = []
    # to track the average evaluation score per epoch
    avg_eval_scores = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=output_path, model_dict=MODEL_DICT)
    # Initialise the One Cycle learning rate scheduler
    lr_scheduler = (torch.optim.lr_scheduler
                    .OneCycleLR(optimizer, max_lr=lr_to_use,
                                steps_per_epoch=len(training_loader),
                                epochs=NUM_EPOCHS, pct_start=0.5))

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        tic = time.perf_counter()
        print("Epoch {} of {}".format(epoch, n_epochs))
        for batch in tqdm(training_loader, desc='Training batch',
                          bar_format=BAR_FORMAT):
            inputs, targets = prepare_batch(batch, DEVICE_NUM)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the
            # model
            output = model(inputs)
            # calculate the loss
            if LOSS_CRITERION == 'CrossEntropyLoss':
                loss = loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = loss_criterion(output, targets)
            # backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update the learning rate
            lr_scheduler.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc='Validation batch',
                              bar_format=BAR_FORMAT):
                inputs, targets = prepare_batch(batch, DEVICE_NUM)
                # forward pass: compute predicted outputs by passing inputs to
                # the model
                output = model(inputs)
                # calculate the loss
                if LOSS_CRITERION == 'CrossEntropyLoss':
                    loss = loss_criterion(output, torch.argmax(targets, dim=1))
                else:
                    loss = loss_criterion(output, targets)
                # record validation loss
                valid_losses.append(loss.item())
                # # if model contains final_activation layer for normalizing
                # # logits apply it, otherwise the evaluation metric will be
                # # incorrectly computed
                # if (hasattr(model, 'final_activation')
                #         and model.final_activation is not None):
                #     output = model.final_activation(output)
                s_max = nn.Softmax(dim=1)
                probs = s_max(output)
                eval_score = eval_criterion(probs, targets)
                eval_scores.append(eval_score)

        toc = time.perf_counter()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        eval_score = np.average(eval_scores)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_eval_scores.append(eval_score)
        print(
            f"Epoch {epoch}. Training loss: {train_loss}, Validation Loss: "
            f"{valid_loss}. {EVAL_METRIC}: {eval_score}")
        print(f"Time taken for epoch {epoch}: {toc - tic:0.2f} seconds")
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        eval_scores = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model_dict = torch.load(output_path, map_location='cpu')
    model.load_state_dict(model_dict['model_state_dict'])

    return model, avg_train_losses, avg_valid_losses, avg_eval_scores


def output_loss_fig_data(train_loss, valid_loss, avg_eval_scores, data_path):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.8)  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_out_pth = data_path/f'{MODEL_OUT_FN[:-8]}_loss_plot.png'
    print(f"Saving figure of training/validation losses to {fig_out_pth}")
    fig.savefig(fig_out_pth, bbox_inches='tight')
    # Output a list of training stats
    epoch_lst = range(len(train_loss))
    rows = zip(epoch_lst, train_loss, valid_loss, avg_eval_scores)
    with open(data_path/f'{MODEL_OUT_FN[:-8]}_train_stats.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Epoch', 'Train Loss', 'Valid Loss', 'Eval Score'))
        for row in rows:
            writer.writerow(row)


def plot_validation_slices(predicted_vol, data_path, filename):
    columns = 3
    rows = 2
    z_dim = predicted_vol.shape[0]//2
    y_dim = predicted_vol.shape[1]//2
    x_dim = predicted_vol.shape[2]//2

    with h5.File(DATA_DIR/VALID_SEG, 'r') as f:
        validation_gt = f['/data'][()]
    fig = plt.figure(figsize=(12, 8))
    pred_z = fig.add_subplot(rows, columns, 1)
    plt.imshow(predicted_vol[z_dim, :, :], cmap='gray')
    pred_y = fig.add_subplot(rows, columns, 2)
    plt.imshow(predicted_vol[:, y_dim, :], cmap='gray')
    pred_x = fig.add_subplot(rows, columns, 3)
    plt.imshow(predicted_vol[:, :, x_dim], cmap='gray')
    gt_z = fig.add_subplot(rows, columns, 4)
    plt.imshow(validation_gt[z_dim, :, :], cmap='gray')
    gt_y = fig.add_subplot(rows, columns, 5)
    plt.imshow(validation_gt[:, y_dim, :], cmap='gray')
    gt_x = fig.add_subplot(rows, columns, 6)
    plt.imshow(validation_gt[:, :, x_dim], cmap='gray')
    pred_z.title.set_text(f'x,y slice pred [{z_dim}, :, :]')
    pred_y.title.set_text(f'z,x slice pred [:, {y_dim}, :]')
    pred_x.title.set_text(f'z,y slice pred [:, :, {x_dim}]')
    gt_z.title.set_text(f'x,y slice GT [{z_dim}, :, :]')
    gt_y.title.set_text(f'z,x slice GT [:, {y_dim}, :]')
    gt_x.title.set_text(f'z,y slice GT [:, :, {x_dim}]')
    plt.suptitle(f"Predictions for {filename}", fontsize=16)
    plt_out_pth = data_path/f'{filename[:-8]}_3d_prediction_images.png'
    print(f"Saving figure of orthogonal slice predictions to {plt_out_pth}")
    plt.savefig(plt_out_pth, dpi=150)


def predict_validation_region(model, validation_dataset, validation_batch_size,
                              thresh_val, data_path):
    sample = validation_dataset[0]
    patch_overlap = 32

    grid_sampler = GridSampler(
        sample,
        PATCH_SIZE,
        patch_overlap,
        padding_mode='reflect'
    )
    patch_loader = DataLoader(
        grid_sampler, batch_size=validation_batch_size)
    aggregator = torchio.data.inference.GridAggregator(grid_sampler)

    model.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['data'][DATA].to(DEVICE_NUM)
            locations = patches_batch[torchio.LOCATION]
            logits = model(inputs)
            s_max = nn.Softmax(dim=1)
            probs = s_max(logits)
            aggregator.add_batch(probs, locations)
    predicted_vol = aggregator.get_output_tensor()  # output is 4D
    print(f"Shape of the predicted Volume is: {predicted_vol.shape}")
    predicted_vol = predicted_vol.numpy().squeeze()  # remove first dimension
    predicted_vol = np.argmax(predicted_vol, axis=0)
    h5_out_path = data_path/f"{MODEL_OUT_FN[:-8]}_validation_vol_predicted.h5"
    print(f"Outputting prediction of the validation volume to {h5_out_path}")
    with h5.File(h5_out_path, 'w') as f:
        f['/data'] = predicted_vol.astype(np.uint8)
    plot_validation_slices(predicted_vol, data_path, MODEL_OUT_FN)


#########################
# Setup data structures #
#########################
def fix_label_classes(data, seg_classes):
    """Changes the data values of classes in a segmented volume so that
    they start from zero.

    Args:
        seg_classes(list): An ascending list of the labels in the volume.
    """
    for idx, current in enumerate(seg_classes):
        data[data == current] = idx


def is_not_consecutive(num_list):
    maximum = max(num_list)
    if sum(num_list) == maximum * (maximum + 1) / 2:
        return False
    return True

# Load the data into tensors
# 2. Check number of labels and fix labels if not consecutive and not starting from 0


train_data = numpy_from_hdf5(TRAIN_DATA)
train_seg = numpy_from_hdf5(TRAIN_SEG)
valid_data = numpy_from_hdf5(VALID_DATA)
valid_seg = numpy_from_hdf5(VALID_SEG)

seg_classes = np.unique(train_seg)
NUM_CLASSES = len(seg_classes)
assert(NUM_CLASSES > 1)
print("Number of classes in segmentation dataset:"
      f" {NUM_CLASSES}")
print(f"These classes are: {seg_classes}")
if (seg_classes[0] != 0) or is_not_consecutive(seg_classes):
    print("Fixing label classes.")
    fix_label_classes(train_seg, seg_classes)
    fix_label_classes(valid_seg, seg_classes)
    print(f"Label classes {np.unique(train_seg)}")
label_codes = [f"label_val_{i}" for i in seg_classes]

# Get the loss function - TODO make this changeable according to number of
# classes in segmentation
if LOSS_CRITERION == 'BCEDiceLoss':
    print(
        f"Using combined BCE and Dice loss with weighting of {ALPHA}*BCE "
        f"and {BETA}*Dice")
    loss_criterion = BCEDiceLoss(ALPHA, BETA)
elif LOSS_CRITERION == 'DiceLoss':
    print("Using DiceLoss")
    loss_criterion = DiceLoss(sigmoid_normalization=False)
elif LOSS_CRITERION == 'BCELoss':
    print("Using BCELoss")
    loss_criterion = nn.BCEWithLogitsLoss()
elif LOSS_CRITERION == 'CrossEntropyLoss':
    print("Using CrossEntropyLoss")
    loss_criterion = nn.CrossEntropyLoss()
elif LOSS_CRITERION == 'GeneralizedDiceLoss':
    print("Using GeneralizedDiceLoss")
    loss_criterion = GeneralizedDiceLoss()
else:
    print("No loss criterion specified, exiting")
    sys.exit(1)
# Get evaluation metric
if EVAL_METRIC == "MeanIoU":
    print("Using MeanIoU")
    eval_criterion = MeanIoU()
elif EVAL_METRIC == "GenericAveragePrecision":
    print("Using GenericAveragePrecision")
    eval_criterion = GenericAveragePrecision()
else:
    print("No evaluation metric specified, exiting")
    sys.exit(1)
# Create model and optimizer
MODEL_DICT['out_channels'] = NUM_CLASSES
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
unet = create_unet_on_device(DEVICE_NUM, MODEL_DICT)
optimizer = torch.optim.AdamW(unet.parameters(), lr=STARTING_LR)

train_subject = torchio.Subject(
    data=torchio.Image(tensor=torch.from_numpy(train_data),
                       label=torchio.INTENSITY),
    label=torchio.Image(tensor=torch.from_numpy(train_seg),
                        label=torchio.LABEL),
)
valid_subject = torchio.Subject(
    data=torchio.Image(tensor=torch.from_numpy(valid_data),
                       label=torchio.INTENSITY),
    label=torchio.Image(tensor=torch.from_numpy(valid_seg),
                        label=torchio.LABEL),
)
# Define the transforms for the set of training patches
training_transform = Compose([
    RandomNoise(p=0.2),
    RandomFlip(axes=(0, 1, 2)),
    RandomBlur(p=0.2),
    OneOf({
        RandomAffine(): 0.8,
        RandomElasticDeformation(): 0.2,
    }, p=0.5),  # Changed from p=0.75 24/6/20
])
# Create the datasets
training_dataset = torchio.ImagesDataset(
    [train_subject], transform=training_transform)

validation_dataset = torchio.ImagesDataset(
    [valid_subject])
# Define the queue of sampled patches for training and validation
sampler = torchio.data.UniformSampler(PATCH_SIZE)
patches_training_set = torchio.Queue(
    subjects_dataset=training_dataset,
    max_length=MAX_QUEUE_LENGTH,
    samples_per_volume=TRAIN_PATCHES,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    shuffle_subjects=False,
    shuffle_patches=True,
)

patches_validation_set = torchio.Queue(
    subjects_dataset=validation_dataset,
    max_length=MAX_QUEUE_LENGTH,
    samples_per_volume=VALID_PATCHES,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    shuffle_subjects=False,
    shuffle_patches=False,
)

total_gpu_mem = torch.cuda.get_device_properties(DEVICE_NUM).total_memory
allocated_gpu_mem = torch.cuda.memory_allocated(DEVICE_NUM)
free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024**3  # free

if free_gpu_mem < 30:
    batch_size = 2  # Set to 2 for 16Gb Card
else:
    batch_size = 2  # Set to 4 for 32Gb Card
print(f"Patch size is {PATCH_SIZE}")
print(f"Free GPU memory is {free_gpu_mem:0.2f} GB. Batch size will be "
      f"{batch_size}.")

training_loader = DataLoader(
    patches_training_set, batch_size=batch_size)

validation_loader = DataLoader(
    patches_validation_set, batch_size=batch_size)

#############
# Do things #
#############
# Create exponential learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
lr_find_loss, lr_find_lr = lr_finder(
    unet, training_loader, optimizer, lr_scheduler)
lr_to_use = find_appropriate_lr(unet, lr_find_loss, lr_find_lr)
print(f"LR to use {lr_to_use}")

# recreate the Unet and the optimizer
print("Recreating the U-net and optimizer")
unet = create_unet_on_device(DEVICE_NUM, MODEL_DICT)
optimizer = torch.optim.AdamW(unet.parameters(), lr=STARTING_LR)

model_out_path = DATA_DIR/MODEL_OUT_FN
output_tuple = train_model(unet, optimizer, lr_to_use, training_loader,
                           validation_loader, loss_criterion, eval_criterion,
                           PATIENCE, NUM_EPOCHS, model_out_path)
unet, avg_train_losses, avg_valid_losses, avg_eval_scores = output_tuple
fig_out_dir = DATA_DIR/f'{date.today()}_3d_train_figs'
os.makedirs(fig_out_dir, exist_ok=True)
output_loss_fig_data(avg_train_losses, avg_valid_losses,
                     avg_eval_scores, fig_out_dir)
predict_validation_region(unet, validation_dataset,
                          batch_size, THRESH_VAL, fig_out_dir)
