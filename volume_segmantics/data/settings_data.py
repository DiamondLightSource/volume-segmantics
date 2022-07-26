import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import yaml


def get_settings_data(data: Union[Path, dict, None]) -> SimpleNamespace:
    """Given a path to a YAML file or a dictionary object, returns a 
    simple namespace object holding settings data. If the data is None, 
    an empty namespace is returned.
    """
    if data is None:
        return SimpleNamespace()
    elif isinstance(data, Path):
        logging.info(f"Loading settings from {data}")
        if data.exists():
            with open(data, "r") as stream:
                settings_dict = yaml.safe_load(stream)
            return SimpleNamespace(**settings_dict)
        else:
            logging.error("Couldn't find settings file... Exiting!")
            sys.exit(1)
    elif isinstance(data, dict):
        return SimpleNamespace(**data)
