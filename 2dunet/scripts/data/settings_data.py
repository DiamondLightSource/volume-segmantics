# -*- coding: utf-8 -*-
"""Data utilities for U-net training and prediction.
"""
import logging
import sys
import warnings

import yaml

warnings.filterwarnings("ignore", category=UserWarning)


class SettingsData:
    """Class to store settings from a YAML settings file.

    Args:
        settings_path (pathlib.Path): Path to the YAML file containing user settings.
    """
    def __init__(self, settings_path):
        logging.info(f"Loading settings from {settings_path}")
        if settings_path.exists():
            self.settings_path = settings_path
            with open(settings_path, 'r') as stream:
                self.settings_dict = yaml.safe_load(stream)
        else:
            logging.error("Couldn't find settings file... Exiting!")
            sys.exit(1)

        # Set the data as attributes
        for k, v in self.settings_dict.items():
            setattr(self, k, v)
