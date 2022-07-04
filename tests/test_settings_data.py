from pathlib import Path
from types import SimpleNamespace

import pytest
from volume_segmantics.data import get_settings_data


def test_get_settings_data_path(training_settings_path):
    settings = get_settings_data(training_settings_path)
    assert isinstance(settings, SimpleNamespace)


def test_get_settings_data_none():
    settings = get_settings_data(None)
    assert isinstance(settings, SimpleNamespace)


def test_get_settings_data_dict():
    settings_dict = {
        "test_data_1": 1234,
        "test_data_2": "A string",
        "test_data_3": {"a_dict": 4.567},
    }
    settings = get_settings_data(settings_dict)
    assert isinstance(settings, SimpleNamespace)
    assert settings.test_data_1 == 1234


def test_get_settings_data_bad_path():
    bad_path = Path("i/have/been/a/very_bad/file")
    with pytest.raises(SystemExit) as wrapped_e:
        settings = get_settings_data(bad_path)
    assert wrapped_e.type == SystemExit
    assert wrapped_e.value.code == 1
