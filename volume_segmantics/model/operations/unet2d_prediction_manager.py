from pathlib import Path

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.data.base_data_manager import BaseDataManager
from volume_segmantics.data.settings_data import SettingsData
from volume_segmantics.model.operations.unet2d_predictor import Unet2dPredictor


class Unet2DPredictionManager(BaseDataManager):
    def __init__(
        self, predictor: Unet2dPredictor, data_vol_path: str, settings: SettingsData
    ) -> None:
        super().__init__(data_vol_path, settings)
        self.predictor = predictor
        self.settings = settings

    def predict_volume_to_path(self, output_path: Path) -> None:
        probs = None
        one_hot = self.settings.one_hot
        quality = utils.get_prediction_quality(self.settings)
        if quality == utils.Quality.LOW:
            if one_hot:
                prediction = self.predictor.predict_single_axis_to_one_hot(
                    self.data_vol
                )
            else:
                prediction, probs = self.predictor.predict_single_axis(self.data_vol)
        if quality == utils.Quality.MEDIUM:
            if one_hot:
                prediction = self.predictor.predict_3_ways_one_hot(self.data_vol)
            else:
                prediction, probs = self.predictor.predict_3_ways_max_probs(
                    self.data_vol
                )
        if quality == utils.Quality.HIGH:
            if one_hot:
                prediction = self.predictor.predict_12_ways_one_hot(self.data_vol)
            else:
                prediction, probs = self.predictor.predict_12_ways_max_probs(
                    self.data_vol
                )
        utils.save_data_to_hdf5(
            prediction, output_path, chunking=self.input_data_chunking
        )
        if probs is not None and self.settings.output_probs:
            utils.save_data_to_hdf5(
                probs,
                f"{output_path.parent / output_path.stem}_probs.h5",
                chunking=self.input_data_chunking,
            )
