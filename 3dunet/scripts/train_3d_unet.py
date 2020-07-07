"""Script to train a 3D-unet for binary segmentation when given data and ground truth
    HDF5 input volumes. A separate set of data and ground truth volumes are required for validation. 
"""
import enum
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
from pytorch3dunet.unet3d.losses import BCEDiceLoss, DiceLoss
from pytorch3dunet.unet3d.metrics import MeanIoU
from pytorch3dunet.unet3d.model import ResidualUNet3D
from torch import nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchio import DATA
from torchio.data.inference import GridSampler
from torchio.transforms import (Compose, OneOf, RandomAffine, RandomBlur,
                                RandomElasticDeformation, RandomFlip,
                                RandomNoise, RescaleIntensity)
from tqdm import tqdm

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)  # Extract the directory of the script
settings_path = Path(dir_path, 'settings', '3d_unet_train_settings.yaml')
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
TRAIN_SEG = DATA_DIR/settings_dict['train_seg']  # Need to ensure values are [0,1]
VALID_DATA = DATA_DIR/settings_dict['valid_data']# Need to ensure values are [0,1]
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
DEVICE_NUM = 0  # Once single GPU slected, default device will always be 0
LOSS_CRITERION = settings_dict['loss_criterion']
ALPHA = settings_dict['alpha']
BETA = settings_dict['beta']

def create_unet_on_device(device, model_dict):
    unet = ResidualUNet3D(**model_dict)
    print(f"Sending the model to device {CUDA_DEVICE}")
    return unet.to(device)

def tensor_from_hdf5(file_path, data_path):
    with h5.File(file_path, 'r') as f:
        tens = torch.from_numpy(f[data_path][()])
    return tens

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
unet = create_unet_on_device(DEVICE_NUM, MODEL_DICT)

# Get the loss function - TODO make this changeable according to number of classes in segmentation
# loss_criterion = nn.BCEWithLogitsLoss()
if LOSS_CRITERION == 'BCEDiceLoss':
    print(f"Using combined BCE and Dice loss with weighting of {ALPHA}*BCE and {BETA}*Dice")
    loss_criterion = BCEDiceLoss(ALPHA, BETA)
elif LOSS_CRITERION == 'DiceLoss':
    print("Using DiceLoss")
    loss_criterion = DiceLoss(ALPHA, BETA)
elif LOSS_CRITERION == 'BCELoss':
    print("Using DiceLoss")
    loss_criterion = nn.BCEWithLogitsLoss()
else:
    print("No loss criterion specified, exiting")
    sys.exit(1)
# Get evaluation metric
eval_criterion = MeanIoU()
# Create optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=STARTING_LR)

# Load the data into tensors
train_data_tens = tensor_from_hdf5(TRAIN_DATA, '/data')
train_seg_tens = tensor_from_hdf5(TRAIN_SEG, '/data')
valid_data_tens = tensor_from_hdf5(VALID_DATA, '/data')
valid_seg_tens = tensor_from_hdf5(VALID_SEG, '/data')

# Ceate a queue with patch based sampling
train_subject = torchio.Subject(
    data=torchio.Image(tensor=train_data_tens, label=torchio.INTENSITY),
    label=torchio.Image(tensor=train_seg_tens, label=torchio.LABEL),
)
valid_subject = torchio.Subject(
    data=torchio.Image(tensor=valid_data_tens, label=torchio.INTENSITY),
    label=torchio.Image(tensor=valid_seg_tens, label=torchio.LABEL),
)

training_transform = Compose([
    RescaleIntensity((0, 1)),
    RandomNoise(p=0.2),
    RandomFlip(axes=(0, 1, 2)),
    RandomBlur(p=0.2),
    OneOf({
        RandomAffine(): 0.8,
        RandomElasticDeformation(): 0.2,
    }, p=0.5),  # Changed from p=0.75 24/6/20
])

training_dataset = torchio.ImagesDataset(
    [train_subject], transform=training_transform)

validation_dataset = torchio.ImagesDataset(
    [valid_subject])


total_gpu_mem = torch.cuda.get_device_properties(DEVICE_NUM).total_memory
allocated_gpu_mem = torch.cuda.memory_allocated(DEVICE_NUM)
free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024**3  # free

if free_gpu_mem < 20:
    training_batch_size = 2  # Set to 4 for 32Gb Card
    validation_batch_size = 2
else:
    training_batch_size = 4  # Set to 4 for 32Gb Card
    validation_batch_size = 4
print(f"Patch size is {PATCH_SIZE}")
print(
    f"Free GPU memory is {free_gpu_mem:0.4f} GB. Batch size will be {training_batch_size}.")

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

training_loader = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

validation_loader = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)


def prepare_batch(batch, device):
    inputs = batch['data'][DATA].to(device)
    targets = batch['label'][DATA].to(device)
    return inputs, targets


# Create learning rate adjustment strategy
# To start with, we want to find the optimum learning rate wilearning_rate As decribed here
# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee

def lr_lambda(x): 
    return math.exp(
    x * math.log(END_LR / STARTING_LR) / (LR_FIND_EPOCHS * len(training_loader))
    )


lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


################## Find a decent learning rate #####################
def lr_finder(model, training_loader, optimizer, lr_scheduler, smoothing=0.05, plt_fig=True):
    lr_find_loss = []
    lr_find_lr = []
    iters = 0

    model.train()
    print(f"Training for {LR_FIND_EPOCHS} epochs to create a learning rate plot.")
    for i in range(LR_FIND_EPOCHS):
        for batch in tqdm(training_loader, desc=f'Epoch {i + 1}, batch number',
                          bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
            inputs, targets = prepare_batch(batch, DEVICE_NUM)
            optimizer.zero_grad()
            output = model(inputs)
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

            iters += 1

    if plt_fig:
        fig = tpl.figure()
        fig.plot(np.log10(lr_find_lr), lr_find_loss, width=50, height=30, xlabel='Log10 Learning Rate')
        fig.show()

    return lr_find_loss, lr_find_lr

def find_appropriate_lr(model, lr_find_loss, lr_find_lr, lr_diff=15, loss_threshold=.05, adjust_value=0.75):
    """Function taken from https://forums.fast.ai/t/automated-learning-rate-suggester/44199
        Parameters:

        lr_diff provides the interval distance by units of the “index of LR” (log transform of LRs) between the right and left bound
        loss_threshold is the maximum difference between the left and right bound’s loss values to stop the shift
        adjust_value is a coefficient to the final learning rate for pure manual adjustment
    """
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(lr_find_loss)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = lr_find_lr

    #Search for index in gradients where loss is lowest before the loss spike
    #Initialize right and left idx using the lr_diff as a spacing unit
    #Set the local min lr as -1 to signify if threshold is too low
    local_min_lr = 0.001  # Add as default value to fix bug
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value

    return lr_to_use

############## Model Training ###################


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    This class taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, optimizer, lr_to_use, training_loader, valid_loader, loss_criterion, eval_criterion,
                patience, n_epochs, output_path):
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
        patience=patience, verbose=True, path=output_path)
    # Initialise the One Cycle learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr_to_use, steps_per_epoch=len(training_loader),
                                                       epochs=NUM_EPOCHS, pct_start=0.5)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        tic = time.perf_counter()
        print("Epoch {} of {}".format(epoch, n_epochs))
        for batch in tqdm(training_loader, desc='Training batch',
                                                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
            inputs, targets = prepare_batch(batch, DEVICE_NUM)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = loss_criterion(output, targets)
            # backward pass: compute gradient of the loss with respect to model parameters
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
                              bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
                inputs, targets = prepare_batch(batch, DEVICE_NUM)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(inputs)
                # calculate the loss
                loss = loss_criterion(output, targets)
                # record validation loss
                valid_losses.append(loss.item())
                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(model, 'final_activation') and model.final_activation is not None:
                    output = model.final_activation(output)
                eval_score = eval_criterion(output, targets)
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
            f'Epoch {epoch}. Training loss: {train_loss}, Validation Loss: {valid_loss}. MeanIOU: {eval_score}')
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
    model.load_state_dict(torch.load(output_path))

    return model, avg_train_losses, avg_valid_losses, avg_eval_scores


def output_loss_fig(train_loss, valid_loss, data_path):
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
    plt.ylim(0, 0.8)  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_out_pth = data_path/'loss_plot.png'
    print(f"Saving figure of training/validation losses to {fig_out_pth}")
    fig.savefig(fig_out_pth, bbox_inches='tight')

def plot_validation_slices(predicted_vol, data_path):
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
    pred_z.title.set_text(f'z slice pred [{z_dim}, :, :]')
    pred_y.title.set_text(f'y slice pred [:, {y_dim}, :]')
    pred_x.title.set_text(f'x slice pred [:, :, {x_dim}]')
    gt_z.title.set_text(f'z slice GT [{z_dim}, :, :]')
    gt_y.title.set_text(f'y slice GT [:, {y_dim}, :]')
    gt_x.title.set_text(f'x slice GT [:, :, {x_dim}]')
    plt.suptitle(f"Predictions for Model", fontsize=16)
    plt_out_pth = data_path/f'3d_prediction_images.png'
    print(f"Saving figure of orthogonal slice predictions to {plt_out_pth}")
    plt.savefig(plt_out_pth, dpi=150)

def predict_validation_region(model, validation_dataset, validation_batch_size, thresh_val, data_path):
    sample = validation_dataset[0]
    patch_overlap = 16

    grid_sampler = GridSampler(
        sample,
        PATCH_SIZE,
        patch_overlap,
        padding_mode='reflect'
    )
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=validation_batch_size)
    aggregator = torchio.data.inference.GridAggregator(grid_sampler)

    model.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['data'][DATA].to(DEVICE_NUM)
            locations = patches_batch[torchio.LOCATION]
            logits = model(inputs)
            logits = model.final_activation(logits)
            aggregator.add_batch(logits, locations)

    predicted_vol = aggregator.get_output_tensor()  # output is 4D
    predicted_vol = predicted_vol.numpy().squeeze() # remove first dimension
    # Threshold
    predicted_vol[predicted_vol >= thresh_val] = 1
    predicted_vol[predicted_vol < thresh_val] = 0
    predicted_vol = predicted_vol.astype(np.uint8)
    h5_out_path = data_path/"validation_vol_predicted.h5"
    print(f"Outputting prediction of the validation volume to {h5_out_path}")
    with h5.File(h5_out_path, 'w') as f:
        f['/data'] = predicted_vol
    plot_validation_slices(predicted_vol, data_path)

########## Do things here ##############
lr_find_loss, lr_find_lr = lr_finder(unet, training_loader, optimizer, lr_scheduler)
lr_to_use = find_appropriate_lr(unet, lr_find_loss, lr_find_lr)
print(f"LR to use {lr_to_use}")

# recreate the Unet and the optimizer
print("Recreating the U-net and optimizer")
unet = create_unet_on_device(DEVICE_NUM, MODEL_DICT)
optimizer = torch.optim.AdamW(unet.parameters(), lr=STARTING_LR)

model_out_path = DATA_DIR/MODEL_OUT_FN
model, avg_train_losses, avg_valid_losses, avg_eval_scores = train_model(unet, optimizer, lr_to_use,
                                                                         training_loader, validation_loader, loss_criterion,
                                                                         eval_criterion, PATIENCE, NUM_EPOCHS, model_out_path)
fig_out_dir = DATA_DIR/f'{date.today()}_3d_training_figs'
os.makedirs(fig_out_dir, exist_ok=True)
output_loss_fig(avg_train_losses, avg_valid_losses, fig_out_dir)
# TODO predict the validation volume
predict_validation_region(unet, validation_dataset,
                          validation_batch_size, THRESH_VAL, fig_out_dir)
