import os
import sys
from pathlib import Path
from datetime import date

import h5py as h5
import numpy as np
import torch
import torchio
import yaml
from matplotlib import pyplot as plt
from pytorch3dunet.unet3d.model import ResidualUNet3D
from torch.utils.data import DataLoader
from torchio.data.inference import GridSampler
from tqdm import tqdm

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)  # Extract the directory of the script
settings_path = Path(dir_path, 'settings', '3d_unet_predict_settings.yaml')
print(f"Loading settings from {settings_path}")
if settings_path.exists():
    with open(settings_path, 'r') as stream:
        settings_dict = yaml.safe_load(stream)
else:
    print("Couldn't find settings file... Exiting!")
    sys.exit(1)

CUDA_DEVICE = str(settings_dict['cuda_device'])
DATA_DIR = Path(settings_dict['data_dir'])
DATA_FILE = DATA_DIR/settings_dict['data_fn']
MODEL_DIR = Path(settings_dict['model_dir'])
MODEL_FILE = MODEL_DIR/settings_dict['model_fn']
DATA_OUT_DIR = Path(settings_dict['data_out_dir'])
DATA_OUT_FN = DATA_OUT_DIR/settings_dict['data_out_fn']
HDF5_PATH = settings_dict['hdf5_path']
PATCH_SIZE = tuple(settings_dict['patch_size'])
PATCH_OVERLAP = settings_dict['patch_overlap']
PADDING_MODE = settings_dict['padding_mode']
MODEL_DICT = settings_dict['model']
THRESH_VAL = settings_dict['thresh_val']
DATA = 'data'
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


def plot_predict_figure(predicted_vol, data_tens, data_path):
    columns = 3
    rows = 2
    z_dim = predicted_vol.shape[0]//2
    y_dim = predicted_vol.shape[1]//2
    x_dim = predicted_vol.shape[2]//2

    fig = plt.figure(figsize=(12, 8))
    pred_z = fig.add_subplot(rows, columns, 1)
    plt.imshow(predicted_vol[z_dim, :, :], cmap='gray')
    pred_y = fig.add_subplot(rows, columns, 2)
    plt.imshow(predicted_vol[:, y_dim, :], cmap='gray')
    pred_x = fig.add_subplot(rows, columns, 3)
    plt.imshow(predicted_vol[:, :, x_dim], cmap='gray')
    gt_z = fig.add_subplot(rows, columns, 4)
    plt.imshow(data_tens[z_dim, :, :].numpy(), cmap='gray')
    gt_y = fig.add_subplot(rows, columns, 5)
    plt.imshow(data_tens[:, y_dim, :].numpy(), cmap='gray')
    gt_x = fig.add_subplot(rows, columns, 6)
    plt.imshow(data_tens[:, :, x_dim].numpy(), cmap='gray')
    pred_z.title.set_text(f'z slice pred [{z_dim}, :, :]')
    pred_y.title.set_text(f'y slice pred [:, {y_dim}, :]')
    pred_x.title.set_text(f'x slice pred [:, :, {x_dim}]')
    gt_z.title.set_text(f'z slice data [{z_dim}, :, :]')
    gt_y.title.set_text(f'y slice data [:, {y_dim}, :]')
    gt_x.title.set_text(f'x slice data [:, :, {x_dim}]')
    plt.suptitle("Predictions for 3d U-net", fontsize=16)
    plt_out_pth = data_path/'3d_prediction_images.png'
    print(f"Saving figure of orthogonal slice predictions to {plt_out_pth}")
    plt.savefig(plt_out_pth, dpi=150)


def predict_volume(model, grid_sampler, batch_size,
                   thresh_val, data_path):

    patch_loader = DataLoader(
        grid_sampler, batch_size=batch_size)
    aggregator = torchio.data.inference.GridAggregator(grid_sampler)

    model.eval()
    with torch.no_grad():
        for patches_batch in tqdm(patch_loader, desc="Predicting Batch",
                                  bar_format=BAR_FORMAT):
            inputs = patches_batch[DATA][DATA].to(DEVICE_NUM)
            locations = patches_batch[torchio.LOCATION]
            logits = model(inputs)
            logits = model.final_activation(logits)
            aggregator.add_batch(logits, locations)

    predicted_vol = aggregator.get_output_tensor()  # output is 4D
    predicted_vol = predicted_vol.numpy().squeeze()  # remove first dimension
    # Threshold
    predicted_vol[predicted_vol >= thresh_val] = 1
    predicted_vol[predicted_vol < thresh_val] = 0
    predicted_vol = predicted_vol.astype(np.uint8)
    print(f"Outputting prediction of the validation volume to {data_path}")
    with h5.File(data_path, 'w') as f:
        f['/data'] = predicted_vol
    return predicted_vol


os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
total_gpu_mem = torch.cuda.get_device_properties(DEVICE_NUM).total_memory
allocated_gpu_mem = torch.cuda.memory_allocated(DEVICE_NUM)
free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024**3  # free

if free_gpu_mem < 30:
    batch_size = 4  # Set to 4 for 16Gb Card
else:
    batch_size = 8  # Set to 8 for 32Gb Card
print(f"Patch size is {PATCH_SIZE}")
print(f"Free GPU memory is {free_gpu_mem:0.2f} GB. Batch size will be "
      f"{batch_size}.")

# Load model
print(f"Loading model from {MODEL_FILE}")
unet = create_unet_on_device(DEVICE_NUM, MODEL_DICT)
unet.load_state_dict(torch.load(MODEL_FILE))
# Load the data and create a sampler
print(f"Loading data from {DATA_FILE}")
data_tens = tensor_from_hdf5(DATA_FILE, HDF5_PATH)
data_subject = torchio.Subject(
    data=torchio.Image(tensor=data_tens, label=torchio.INTENSITY)
)
print(f"Setting up grid sampler with overlap {PATCH_OVERLAP} and padding "
      f"mode: {PADDING_MODE}")
grid_sampler = GridSampler(
    data_subject,
    PATCH_SIZE,
    PATCH_OVERLAP,
    padding_mode=PADDING_MODE
)

pred_vol = predict_volume(unet, grid_sampler, batch_size,
                          THRESH_VAL, DATA_OUT_FN)
fig_out_dir = DATA_OUT_DIR/f'{date.today()}_3d_prediction_figs'
print(f"Creating directory for figures: {fig_out_dir}")
os.makedirs(fig_out_dir, exist_ok=True)
plot_predict_figure(pred_vol, data_tens, fig_out_dir)
