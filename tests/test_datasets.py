import numpy as np
import pytest
import torch
from volume_segmantics.data.datasets import (
    VolSeg2dDataset,
    VolSeg2dPredictionDataset,
    get_2d_prediction_dataset,
    get_2d_training_dataset,
    get_2d_validation_dataset,
)

@pytest.fixture()
def training_dataset(image_dir, label_dir, training_settings):
    return get_2d_training_dataset(image_dir, label_dir, training_settings)

@pytest.fixture()
def validation_dataset(image_dir, label_dir, training_settings):
    return get_2d_validation_dataset(image_dir, label_dir, training_settings)

@pytest.fixture()
def prediction_dataset(rand_int_volume):
    return get_2d_prediction_dataset(rand_int_volume)

class Test2dDataset:
    def test_get_2d_training_dataset_type(self, training_dataset):
        assert isinstance(training_dataset, VolSeg2dDataset)

    def test_get_2d_training_dataset_length(self, training_dataset):
        assert len(training_dataset) == 20

    def test_get_2d_training_dataset_get_item(self, training_dataset):
        im, mask = training_dataset[2]
        assert isinstance(im, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_get_2d_validation_dataset_type(self, validation_dataset):
        assert isinstance(validation_dataset, VolSeg2dDataset)

    def test_get_2d_validation_dataset_length(self, validation_dataset):
        assert len(validation_dataset) == 20

    def test_get_2d_validation_dataset_get_item(self, validation_dataset):
        im, mask = validation_dataset[2]
        assert isinstance(im, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

class Test2dPredictionDataset:
    def test_get_2d_prediction_dataset_type(self, prediction_dataset):
        assert isinstance(prediction_dataset, VolSeg2dPredictionDataset)

    def test_get_2d_prediction_dataset_length(self, prediction_dataset, rand_int_volume):
        assert len(prediction_dataset) == rand_int_volume.shape[0]

    def test_get_2d_prediction_dataset_get_item(self, prediction_dataset):
        im = prediction_dataset[2]
        assert isinstance(im, torch.Tensor)
