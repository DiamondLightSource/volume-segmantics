from pathlib import Path

import h5py as h5
import numpy as np
import pytest
import volume_segmantics.utilities.config as cfg
from volume_segmantics.data import get_settings_data
from imageio import volwrite


def del_dir(target):
    """
    Delete a given directory and its subdirectories.

    :param target: The directory to delete
    """
    target = Path(target).expanduser()
    assert target.is_dir()
    for p in sorted(target.glob("**/*"), reverse=True):
        if not p.exists():
            continue
        p.chmod(0o666)
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()
    target.rmdir()


@pytest.fixture()
def empty_dir(tmp_path):
    tmp_dir = tmp_path / "empty_dir"
    tmp_dir.mkdir(exist_ok=True)
    yield tmp_dir
    del_dir(tmp_dir)


@pytest.fixture()
def cwd():
    return Path(__file__).parent


@pytest.fixture()
def training_settings(cwd):
    settings_path = Path(cwd.parent, "settings", cfg.TRAIN_SETTINGS_FN)
    training_settings = get_settings_data(settings_path)
    return training_settings


@pytest.fixture()
def prediction_settings(cwd):
    settings_path = Path(cwd.parent, "settings", cfg.PREDICTION_SETTINGS_FN)
    return get_settings_data(settings_path)


@pytest.fixture()
def rand_size():
    return np.random.randint(2, 256, size=(3))


@pytest.fixture()
def rand_int_volume(rand_size):
    return np.random.randint(256, size=rand_size)


@pytest.fixture()
def rand_float_volume(rand_size):
    return np.random.uniform(-1, 1, size=rand_size)


@pytest.fixture()
def rand_float_nan_volume(rand_float_volume):
    min_dim_length = np.min(rand_float_volume.shape)
    rand_coord = np.random.randint(min_dim_length, size=(3))
    rand_float_volume[rand_coord] = np.nan
    return rand_float_volume


@pytest.fixture()
def rand_int_hdf5_path(tmp_path, rand_int_volume, training_settings):
    output_path = tmp_path / "random_int_vol.h5"
    with h5.File(output_path, "w") as f:
        f[training_settings.data_hdf5_path] = rand_int_volume
    return output_path

@pytest.fixture()
def rand_int_tiff_path(tmp_path, rand_int_volume):
    output_path = tmp_path / "random_int_vol.tiff"
    volwrite(output_path, rand_int_volume)
    return output_path
