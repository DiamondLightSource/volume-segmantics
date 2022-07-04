from pathlib import Path

import h5py as h5
import numpy as np
import pytest
import volume_segmantics.utilities.config as cfg
from imageio import volwrite
from skimage import img_as_ubyte, io
from volume_segmantics.data import get_settings_data


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
def training_settings_path(cwd):
    return Path(cwd.parent, "settings", cfg.TRAIN_SETTINGS_FN)


@pytest.fixture()
def prediction_settings_path(cwd):
    return Path(cwd.parent, "settings", cfg.PREDICTION_SETTINGS_FN)


@pytest.fixture()
def training_settings(training_settings_path):
    return get_settings_data(training_settings_path)


@pytest.fixture()
def prediction_settings(prediction_settings_path):
    return get_settings_data(prediction_settings_path)


@pytest.fixture()
def rand_size():
    return np.random.randint(2, 256, size=(3))


@pytest.fixture()
def rand_int_volume(rand_size):
    return np.random.randint(256, size=rand_size)


@pytest.fixture()
def rand_label_volume(rand_size):
    return np.random.randint(4, size=rand_size)


@pytest.fixture()
def rand_label_volume_no_zeros(rand_size):
    return np.random.randint(1, 5, size=rand_size)


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
    yield output_path
    output_path.unlink()


@pytest.fixture()
def rand_label_hdf5_path(tmp_path, rand_label_volume, training_settings):
    output_path = tmp_path / "random_label_vol.h5"
    with h5.File(output_path, "w") as f:
        f[training_settings.seg_hdf5_path] = rand_label_volume
    yield output_path
    output_path.unlink()


@pytest.fixture()
def rand_int_tiff_path(tmp_path, rand_int_volume):
    output_path = tmp_path / "random_int_vol.tiff"
    volwrite(output_path, rand_int_volume)
    yield output_path
    output_path.unlink()


@pytest.fixture()
def rand_label_tiff_path(tmp_path, rand_label_volume):
    output_path = tmp_path / "random_label_vol.tiff"
    volwrite(output_path, rand_label_volume)
    yield output_path
    output_path.unlink()


@pytest.fixture()
def image_dir(empty_dir):
    dir_path = empty_dir / "data"
    dir_path.mkdir(exist_ok=True)
    for i in range(20):
        im = np.random.randint(256, size=(243, 345)).astype(np.uint8)
        path = dir_path / f"data_z_stack_{i}"
        io.imsave(f"{path}.png", im, check_contrast=False)
    yield dir_path
    del_dir(dir_path)


@pytest.fixture()
def label_dir(empty_dir):
    dir_path = empty_dir / "seg"
    dir_path.mkdir(exist_ok=True)
    for i in range(20):
        im = np.random.randint(4, size=(243, 345)).astype(np.uint8)
        path = dir_path / f"seg_z_stack_{i}"
        io.imsave(f"{path}.png", im, check_contrast=False)
    yield dir_path
    del_dir(dir_path)
