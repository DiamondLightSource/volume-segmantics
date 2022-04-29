import logging
from pathlib import Path

from utilities.settingsdata import SettingsData
from utilities.unet2d.predictor import Unet2dPredictor
from utilities.base_data_manager import BaseDataManager


class Unet2DPredictionManager(BaseDataManager):
    def __init__(
        self, predictor: Unet2dPredictor, data_vol_path: str, settings: SettingsData
    ) -> None:
        super().__init__(data_vol_path, settings)
        self.input_data_chunking = None
        self.predictor = predictor
        self.settings = settings

    def predict_volume(self) -> None:
        quality = self.settings.quality
        if quality == "low":
            prediction = self.predictor.predict_single_axis(self.data_vol)
