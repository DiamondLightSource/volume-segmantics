import math
from pathlib import Path

import numpy as np
import pytest
from volume_segmantics.data.base_data_manager import BaseDataManager


@pytest.fixture()
def base_dm_path(rand_int_hdf5_path, training_settings):
    return BaseDataManager(rand_int_hdf5_path, training_settings)


@pytest.fixture()
def base_dm_int_vol(rand_int_volume, training_settings):
    return BaseDataManager(rand_int_volume, training_settings)


@pytest.fixture()
def base_dm_float_vol(rand_float_volume, training_settings):
    return BaseDataManager(rand_float_volume, training_settings)


class TestBaseDataManagerInit:
    def test_init_path_type(self, base_dm_path):
        assert isinstance(base_dm_path, BaseDataManager)

    def test_init_path_data_mean(self, base_dm_path):
        assert base_dm_path.data_mean is not None

    def test_init_path_attribute_path(self, base_dm_path):
        assert isinstance(base_dm_path.data_vol_path, Path)

    def test_init_path_array_exists(self, base_dm_path):
        assert isinstance(base_dm_path.data_vol, np.ndarray)

    def test_init_path_tiff_array_exists(self, rand_int_tiff_path, training_settings):
        base_dm = BaseDataManager(rand_int_tiff_path, training_settings)
        assert isinstance(base_dm.data_vol, np.ndarray)

    def test_init_int_vol_type(self, base_dm_int_vol):
        assert isinstance(base_dm_int_vol, BaseDataManager)

    def test_init_int_vol_data_mean(self, base_dm_int_vol):
        assert base_dm_int_vol.data_mean is not None

    def test_init_int_vol_attribute_path(self, base_dm_int_vol):
        assert base_dm_int_vol.data_vol_path is None

    def test_init_int_vol_array_exists(self, base_dm_int_vol):
        assert isinstance(base_dm_int_vol.data_vol, np.ndarray)

    def test_init_float_vol_data_mean(self, base_dm_float_vol):
        assert base_dm_float_vol.data_mean is not None

    def test_init_float_vol_array_exists(self, base_dm_float_vol):
        assert isinstance(base_dm_float_vol.data_vol, np.ndarray)


class TestBaseDataManagerPreprocessData:
    def test_preprocess_downsampled(
        self, rand_int_hdf5_path, training_settings, rand_int_volume
    ):
        training_settings.downsample = True
        base_dm = BaseDataManager(rand_int_hdf5_path, training_settings)
        assert base_dm.data_vol.shape[0] == math.ceil(rand_int_volume.shape[0] / 2)
        assert base_dm.data_vol.shape[1] == math.ceil(rand_int_volume.shape[1] / 2)
        assert base_dm.data_vol.shape[2] == math.ceil(rand_int_volume.shape[2] / 2)

    def test_preprocess_clip_int(self, training_settings, rand_int_volume):
        training_settings.clip_data = True
        base_dm = BaseDataManager(rand_int_volume, training_settings)
        assert base_dm.data_vol.dtype == np.uint16

    def test_preprocess_clip_uint8(self, training_settings, rand_uint8_volume):
        training_settings.clip_data = True
        base_dm = BaseDataManager(rand_uint8_volume, training_settings)
        assert base_dm.data_vol.dtype == np.uint8

    def test_preprocess_clip_float(self, rand_float_volume, training_settings):
        training_settings.clip_data = True
        base_dm = BaseDataManager(rand_float_volume, training_settings)
        assert base_dm.data_vol.dtype == np.uint16

    def test_preprocess_replace_nan(
        self,
        rand_float_nan_volume,
        training_settings,
    ):
        assert np.isnan(rand_float_nan_volume).any()
        base_dm = BaseDataManager(rand_float_nan_volume, training_settings)
        assert not np.isnan(base_dm.data_vol).any()

    def test_preprocess_replace_nan_clip(
        self,
        rand_float_nan_volume,
        training_settings,
    ):
        assert np.isnan(rand_float_nan_volume).any()
        training_settings.clip_data = True
        base_dm = BaseDataManager(rand_float_nan_volume, training_settings)
        assert not np.isnan(base_dm.data_vol).any()
