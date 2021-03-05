import logging

import h5py as h5
import numpy as np
import torch
import torchio
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.utils.data import DataLoader
from torchio.data.inference import GridAggregator, GridSampler
from torchio.transforms import (
    Compose,
    OneOf,
    RandomAffine,
    RandomBlur,
    RandomElasticDeformation,
    RandomFlip,
    RandomNoise,
    RescaleIntensity,
)
from tqdm import tqdm

from . import config as cfg
from .data_utils_base import BaseDataUtils

DATA = "data"
LOCATION = "location"
DEVICE_NUM = 0  # Once single GPU slected, default device will always be 0
BAR_FORMAT = "{l_bar}{bar: 30}{r_bar}{bar: -30b}"  # tqdm progress bar


class Base3dSampler(BaseDataUtils):
    def __init__(self, settings):
        super().__init__(settings)
        self.patch_size = tuple(settings.patch_size)
        self.patch_overlap = settings.patch_overlap

    def get_batch_size(self):
        total_gpu_mem = torch.cuda.get_device_properties(DEVICE_NUM).total_memory
        allocated_gpu_mem = torch.cuda.memory_allocated(DEVICE_NUM)
        free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024 ** 3  # free

        if free_gpu_mem < 30:
            batch_size = 2  # Set to 2 for 16Gb Card
        else:
            batch_size = 3  # Set to 3 for 32Gb Card
        logging.info(
            f"Free GPU memory is {free_gpu_mem:0.2f} GB. Batch size will be "
            f"{batch_size}."
        )
        return batch_size

    def predict_volume(self, model, data_vol, data_out_path):
        batch_size = self.get_batch_size()
        data_subject = torchio.Subject(
            data=torchio.Image(
                tensor=torch.from_numpy(data_vol), label=torchio.INTENSITY
            )
        )
        logging.info(
            f"Setting up GridSampler, patch size is {self.patch_size}, overlap is {self.patch_overlap}."
        )
        grid_sampler = GridSampler(
            data_subject, self.patch_size, self.patch_overlap, padding_mode="reflect"
        )

        patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = GridAggregator(grid_sampler)

        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(
                patch_loader, desc="Predicting Batch", bar_format=BAR_FORMAT
            ):
                inputs = patches_batch[DATA][DATA].to(DEVICE_NUM)
                locations = patches_batch[LOCATION]
                logits = model(inputs)
                s_max = nn.Softmax(dim=1)
                probs = s_max(logits)
                aggregator.add_batch(probs, locations)

        predicted_vol = aggregator.get_output_tensor()  # output is 4D
        predicted_vol = predicted_vol.numpy().squeeze()  # remove first dimension
        predicted_vol = np.argmax(predicted_vol, axis=0)  # flatten channels
        logging.info(f"Outputting prediction of the volume to {data_out_path}")
        with h5.File(data_out_path, "w") as f:
            f["/data"] = predicted_vol
        return predicted_vol

    def plot_predict_figure(self, input_vol, predicted_vol, output_path, validation=False):
        columns = 3
        rows = 2
        z_dim = predicted_vol.shape[0] // 2
        y_dim = predicted_vol.shape[1] // 2
        x_dim = predicted_vol.shape[2] // 2

        if validation:
            comparison_str = "GT"
        else:
            comparison_str = "data"
        fig = plt.figure(figsize=(12, 8))
        pred_z = fig.add_subplot(rows, columns, 1)
        plt.imshow(predicted_vol[z_dim, :, :], cmap="gray")
        pred_y = fig.add_subplot(rows, columns, 2)
        plt.imshow(predicted_vol[:, y_dim, :], cmap="gray")
        pred_x = fig.add_subplot(rows, columns, 3)
        plt.imshow(predicted_vol[:, :, x_dim], cmap="gray")
        gt_z = fig.add_subplot(rows, columns, 4)
        plt.imshow(input_vol[z_dim, :, :], cmap="gray")
        gt_y = fig.add_subplot(rows, columns, 5)
        plt.imshow(input_vol[:, y_dim, :], cmap="gray")
        gt_x = fig.add_subplot(rows, columns, 6)
        plt.imshow(input_vol[:, :, x_dim], cmap="gray")
        pred_z.title.set_text(f"x,y slice pred [{z_dim}, :, :]")
        pred_y.title.set_text(f"z,x slice pred [:, {y_dim}, :]")
        pred_x.title.set_text(f"z,y slice pred [:, :, {x_dim}]")
        gt_z.title.set_text(f"x,y slice {comparison_str} [{z_dim}, :, :]")
        gt_y.title.set_text(f"z,x slice {comparison_str} [:, {y_dim}, :]")
        gt_x.title.set_text(f"z,y slice {comparison_str} [:, :, {x_dim}]")
        plt.suptitle("Predictions for 3d U-net", fontsize=16)
        logging.info(f"Saving figure of orthogonal slice predictions to {output_path}")
        plt.savefig(output_path, dpi=150)


class TrainingData3dSampler(Base3dSampler):
    def __init__(self, settings):
        super().__init__(settings)
        self.data_vol = self.load_in_vol(settings, cfg.TRAIN_DATA_ARG)
        self.seg_vol = self.load_in_vol(settings, cfg.LABEL_DATA_ARG)
        self.data_val_vol = self.load_in_vol(settings, cfg.TRAIN_VAL_DATA_ARG)
        self.seg_val_vol = self.load_in_vol(settings, cfg.LABEL_VAL_DATA_ARG)
        seg_classes = np.unique(self.seg_vol)
        self.num_seg_classes = len(seg_classes)
        if self.num_seg_classes > 2:
            self.multilabel = True
        logging.info(
            "Number of classes in segmentation dataset:" f" {self.num_seg_classes}"
        )
        logging.info(f"These classes are: {seg_classes}")
        if (seg_classes[0] != 0) or self.is_not_consecutive(seg_classes):
            logging.info("Fixing label classes.")
            self.fix_label_classes(self.seg_vol, seg_classes)
            self.fix_label_classes(self.seg_val_vol, seg_classes)
        self.create_volume_data_loaders(settings)

    def get_training_loader(self):
        return self.training_loader

    def get_validation_loader(self):
        return self.validation_loader

    def create_volume_data_loaders(self, settings):
        train_subject = torchio.Subject(
            data=torchio.Image(
                tensor=torch.from_numpy(self.data_vol), label=torchio.INTENSITY
            ),
            label=torchio.Image(
                tensor=torch.from_numpy(self.seg_vol), label=torchio.LABEL
            ),
        )
        valid_subject = torchio.Subject(
            data=torchio.Image(
                tensor=torch.from_numpy(self.data_val_vol), label=torchio.INTENSITY
            ),
            label=torchio.Image(
                tensor=torch.from_numpy(self.seg_val_vol), label=torchio.LABEL
            ),
        )
        # Define the transforms for the set of training patches
        training_transform = Compose(
            [
                RandomNoise(p=0.2),
                RandomFlip(axes=(0, 1, 2)),
                RandomBlur(p=0.2),
                OneOf(
                    {
                        RandomAffine(): 0.8,
                        RandomElasticDeformation(): 0.2,
                    },
                    p=0.5,
                ),  # Changed from p=0.75 24/6/20
            ]
        )
        # Create the datasets
        training_dataset = torchio.ImagesDataset(
            [train_subject], transform=training_transform
        )

        validation_dataset = torchio.ImagesDataset([valid_subject])
        # Define the queue of sampled patches for training and validation
        sampler = torchio.data.UniformSampler(tuple(settings.patch_size))
        patches_training_set = torchio.Queue(
            subjects_dataset=training_dataset,
            max_length=settings.max_queue_length,
            samples_per_volume=settings.train_patches,
            sampler=sampler,
            num_workers=settings.num_workers,
            shuffle_subjects=False,
            shuffle_patches=True,
        )

        patches_validation_set = torchio.Queue(
            subjects_dataset=validation_dataset,
            max_length=settings.max_queue_length,
            samples_per_volume=settings.valid_patches,
            sampler=sampler,
            num_workers=settings.num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
        allocated_gpu_mem = torch.cuda.memory_allocated(0)
        free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024 ** 3  # free

        if free_gpu_mem < 30:
            batch_size = 2  # Set to 2 for 16Gb Card
        else:
            batch_size = 3  # Set to 3 for 32Gb Card
        logging.info(f"Patch size is {tuple(settings.patch_size)}")
        logging.info(
            f"Free GPU memory is {free_gpu_mem:0.2f} GB. Batch size will be "
            f"{batch_size}."
        )

        self.training_loader = DataLoader(patches_training_set, batch_size=batch_size)
        self.validation_loader = DataLoader(
            patches_validation_set, batch_size=batch_size
        )

class PredictionData3dSampler(Base3dSampler):
    def __init__(self, settings, predictor):
        super().__init__(settings)
        self.data_vol = self.load_in_vol(settings, cfg.PREDICT_DATA_ARG)
