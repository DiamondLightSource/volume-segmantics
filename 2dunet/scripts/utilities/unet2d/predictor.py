import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm
import utilities.config as cfg
from utilities.dataloaders import get_2d_prediction_dataloader
from utilities.settingsdata import SettingsData
from utilities.unet2d.model import create_unet_from_file


class Unet2dPredictor:
    """Class that performs U-Net prediction operations. Does not interact with disk."""

    def __init__(self, model_file_path: str, settings: SettingsData) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        self.model, self.num_labels = create_unet_from_file(self.model_file_path, self.model_device_num)

    def get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def predict_single_axis(self, data_vol):
        output_vol_list = []
        s_max = nn.Softmax(dim=1)
        data_loader = get_2d_prediction_dataloader(data_vol, self.settings)
        self.model.eval()
        logging.info(f"Predicting segmentation for volume of shape {data_vol.shape}.")
        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc="Prediction batch",
                bar_format=cfg.TQDM_BAR_FORMAT
            ):
                #print(batch.shape)
                output = self.model(batch.to(self.model_device_num))  # Forward pass
                probs = s_max(output)  # Convert the logits to probs
                labels = torch.argmax(probs, dim=1)  # flatten channels
                if labels.is_cuda:
                    labels = labels.cpu()
                labels = labels.detach().numpy()
                labels = labels.astype(np.uint8)
                output_vol_list.append(labels)
        return np.concatenate(output_vol_list)
