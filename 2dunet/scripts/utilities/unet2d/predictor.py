import logging
from pathlib import Path

from utilities.settingsdata import SettingsData
from utilities.unet2d.model import create_unet_from_file


class Unet2dPredictor:
    """Class that performs U-Net prediction operations. Does not interact with disk."""

    def __init__(self, model_file_path: str, settings: SettingsData) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        self.model = create_unet_from_file(self.model_file_path, self.model_device_num)

    def get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def predict_single_axis(self, data_vol):
        data_loader = 0
