"""Script to train a 3D-unet for binary segmentation when given data and ground truth
    HDF5 input volumes. A separate set of data and ground truth volumes are required for validation. 
"""
import enum
import math
import os
import time
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import torchio
from torch import nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchio import DATA
from torchio.transforms import (Compose, OneOf, RandomAffine, RandomBlur,
                                RandomElasticDeformation, RandomFlip,
                                RandomNoise, RescaleIntensity)
from tqdm import tqdm

import termplotlib as tpl
from pytorch3dunet.unet3d.metrics import MeanIoU
from pytorch3dunet.unet3d.model import ResidualUNet3D

CUDA_DEVICE = '2' # Select a particular GPU
STARTING_LR = 1e-6
END_LR = 0.1
LR_FIND_EPOCHS = 2
DATA_DIR = Path('/dls/i12/data/2019/nt23252-1/processing/olly/200608_3d_unet_development')
TRAIN_DATA = 'data_vol_train_384_image.h5' 
TRAIN_SEG =  'seg_vol_train_384_image.h5' # Need to ensure values are [0,1]
VALID_DATA = 'data_vol_valid_256_image_uint8.h5'
VALID_SEG = 'seg_vol_valid_256_image.h5' # Need to ensure values are [0,1]
PATCH_SIZE = (128, 128, 128)
SAMPLE_PER_VOL = 48
MAX_QUEUE_LENGTH = 48
NUM_WORKERS = 8
MODEL_OUT_FN = '200626_3dUnet_vol3_blood_tiss_ADAMW_50_best.pytorch'
NUM_EPOCHS = 50

device = 0  # Once single GPU slected, default device will always be 0
ds_data = 'data'  # Keys for accesi
ds_label = 'label'

model_dict = {
    # model class, e.g. UNet3D, ResidualUNet3D
  "name": "ResidualUNet3D",
  # number of input channels to the model
  "in_channels": 1,
  # number of output channels
  "out_channels": 1,
  # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
  "layer_order": "gcr",
  # feature maps scale factor
  "f_maps": 32,
  # number of groups in the groupnorm
  "num_groups": 8,
  # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
  "final_sigmoid": True,
  # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
  "is_segmentation": True
}

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE

def create_unet_on_device(device, model_dict):
    unet = ResidualUNet3D(**model_dict)
    print(f"Sending the model to device {CUDA_DEVICE}")
    return unet.to(device)

def tensor_from_hdf5(file_path, data_path):
    with h5.File(file_path, 'r') as f:
        tens = torch.from_numpy(f[data_path][()])
    return tens

unet = create_unet_on_device(device, model_dict)

# Get the loss function - TODO make this changable accroding to number of classes in segmentation
loss_criterion = nn.BCEWithLogitsLoss()
# Get evaluation metric
eval_criterion = MeanIoU()
# Create optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=STARTING_LR)

#Load the data into tensors
train_data_tens = tensor_from_hdf5(DATA_DIR/TRAIN_DATA, '/data')
train_seg_tens = tensor_from_hdf5(DATA_DIR/TRAIN_SEG, '/data')
valid_data_tens = tensor_from_hdf5(DATA_DIR/VALID_DATA, '/data')
valid_seg_tens = tensor_from_hdf5(DATA_DIR/VALID_SEG, '/data')

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
    }, p=0.5), # Changed from p=0.75 24/6/20
])

training_dataset = torchio.ImagesDataset(
    [train_subject], transform=training_transform)

validation_dataset = torchio.ImagesDataset(
    [valid_subject])


total_gpu_mem = torch.cuda.get_device_properties(device).total_memory
allocated_gpu_mem = torch.cuda.memory_allocated(device)
free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024**3  # free 

if free_gpu_mem < 20: 
    training_batch_size = 2 # Set to 4 for 32Gb Card
    validation_batch_size = 2
else:
    training_batch_size = 4 # Set to 4 for 32Gb Card
    validation_batch_size = 4
print(f"Patch size is {PATCH_SIZE}")
print(f"Free GPU memory is {free_gpu_mem:0.4f} GB. Batch size will be {training_batch_size}.")

sampler = torchio.data.UniformSampler(PATCH_SIZE)

patches_training_set = torchio.Queue(
    subjects_dataset=training_dataset,
    max_length=MAX_QUEUE_LENGTH,
    samples_per_volume=SAMPLE_PER_VOL,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    shuffle_subjects=False,
    shuffle_patches=True,
)

patches_validation_set = torchio.Queue(
    subjects_dataset=validation_dataset,
    max_length=MAX_QUEUE_LENGTH,
    samples_per_volume=SAMPLE_PER_VOL,
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
    inputs = batch[ds_data][DATA].to(device)
    targets = batch[ds_label][DATA].to(device)
    return inputs, targets


# Create learning rate adjustment strategy
# To start with, we want to find the optimum learning rate wilearning_rate As decribed here 
# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee

lr_lambda = lambda x: math.exp(x * math.log(END_LR / STARTING_LR) / (LR_FIND_EPOCHS * len(training_loader)))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


################## Find a decent learning rate #####################

unet.train()
lr_find_loss = []
lr_find_lr = []
iters = 0
smoothing = 0.05
print(f"Training for {LR_FIND_EPOCHS} epochs to create a learning rate plot.")
for i in range(LR_FIND_EPOCHS):   
    for batch_idx, batch in enumerate(tqdm(training_loader, desc=f'Epoch {i + 1}, batch number',
                                                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        output = unet(inputs)
        loss = loss_criterion(output, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        lr_find_lr.append(lr_step)
        if iters==0:
            lr_find_loss.append(loss)
        else:
            loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]
            lr_find_loss.append(loss)
     
        iters += 1


fig = tpl.figure()
fig.plot(lr_find_lr, lr_find_loss, width=50, height=30, xlabel='Learning Rate')
fig.show()

# TODO Wrap this in a funnction 

lr_diff = 15
loss_threshold = .05
losses = np.array(lr_find_loss)
assert(lr_diff < len(losses))
loss_grad = np.gradient(losses)
lrs = lr_find_lr
    
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

lr_to_use = local_min_lr * 0.75
print(f"LR to use {lr_to_use}")

############## Model Training ###################
 # recreate the Unet and the optimizer
print("Recreating the U-net and optimizer")
unet = create_unet_on_device(device, model_dict)
optimizer = torch.optim.AdamW(unet.parameters(), lr=STARTING_LR)


# Now train with 1 cycle policy with max LR set between to lr_to_use
# TODO - Put this into a function
best_eval = 0
model_out_path = DATA_DIR/MODEL_OUT_FN
train_loss_lst = []
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr_to_use, steps_per_epoch=len(training_loader), epochs=NUM_EPOCHS, pct_start=0.5)
for i in range(NUM_EPOCHS):
    unet.train()
    tic = time.perf_counter()
    print("epoch {} of {}".format(i + 1, NUM_EPOCHS))
    for batch_idx, batch in enumerate(tqdm(training_loader, desc='Training batch',
                                                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
        inputs, targets = prepare_batch(batch, device)
        output = unet(inputs)
        loss = loss_criterion(output, targets)
        train_loss_lst.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    # Validate
    val_loss_lst = []
    val_eval_lst = []
    unet.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_loader, desc='Validation batch',
                                                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
            inputs, targets = prepare_batch(batch, device)
            output = unet(inputs)
            val_loss = loss_criterion(output, targets)
            val_loss_lst.append(val_loss.item())
            # if model contains final_activation layer for normalizing logits apply it, otherwise
            # the evaluation metric will be incorrectly computed
            if hasattr(unet, 'final_activation') and unet.final_activation is not None:
                output = unet.final_activation(output)
            eval_score = eval_criterion(output, targets)
            val_eval_lst.append(eval_score.item())
            eval_avg = np.average(val_eval_lst)
        print(f'Epoch {i + 1}. Training loss: {np.average(train_loss_lst)}, Validation Loss: {np.average(val_loss_lst)}. MeanIOU: {eval_avg}')
        if eval_avg > best_eval:
            best_eval = eval_avg
            state_dict = unet.state_dict()
            print(f'Saving best model with MeanIOU of {eval_avg} to {model_out_path}')
            torch.save(state_dict, model_out_path)
            
    toc = time.perf_counter()
    print(f"Time taken for epoch {i + 1}: {toc - tic:0.4f} seconds")