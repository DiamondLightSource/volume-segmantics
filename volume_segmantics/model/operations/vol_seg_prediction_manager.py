from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.data.base_data_manager import BaseDataManager
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor


class VolSeg2DPredictionManager(BaseDataManager):
    def __init__(
        self,
        predictor: VolSeg2dPredictor,
        data_vol: Union[str, np.ndarray],
        settings: SimpleNamespace,
    ) -> None:
        super().__init__(data_vol, settings)
        self.predictor = predictor
        self.settings = settings

    def predict_volume_to_path(
        self, output_path: Union[Path, None], quality: Union[utils.Quality, None] = None
    ) -> np.ndarray:
        probs = None
        one_hot = self.settings.one_hot
        if quality is None:
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
        if output_path is not None:
            utils.save_data_to_hdf5(
                prediction, output_path, chunking=self.input_data_chunking
            )
            if probs is not None and self.settings.output_probs:
                utils.save_data_to_hdf5(
                    probs,
                    f"{output_path.parent / output_path.stem}_probs.h5",
                    chunking=self.input_data_chunking,
                )
        return prediction
