import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import yaml


def get_settings_data(path: Union[Path, None]) -> SimpleNamespace:
    """Creates an object to hold settings data. If a path to to a YAML file,
    the settings are read in from that file. If the path is None, an empty namespace
    is returned.
    """
    if path is None:
        return SimpleNamespace()
    else:
        logging.info(f"Loading settings from {path}")
        if path.exists():
            with open(path, "r") as stream:
                settings_dict = yaml.safe_load(stream)
            return SimpleNamespace(**settings_dict)
        else:
            logging.error("Couldn't find settings file... Exiting!")
            sys.exit(1)
