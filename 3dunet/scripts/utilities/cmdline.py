import argparse
from pathlib import Path


def CheckExt(choices):
    """Wrapper to return the class
    """
    class Act(argparse.Action):
        """Class to allow checking of filename extensions in argparse. Also 
        checks whether file exists. Adapted from 
        https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
        """
        def __call__(self, parser, namespace, fnames, option_string=None):
            # Modified to take in a list of filenames
            if isinstance(fnames, list):
                for fname in fnames:
                    self.check_path(parser, fname)
            else:
                self.check_path(parser, fnames)
            
            # If all okay, set attribute
            setattr(namespace, self.dest, fnames)

        def check_path(self, parser, fname):
            fname = Path(fname)
            ext = fname.suffix
            if ext not in choices:
                parser.error(f"Wrong filetype: file {fname} doesn't end with {choices}")
                    # Check that file exists
            if not fname.is_file():
                parser.error(f"The file {str(fname)} does not appear to exist.")

    return Act
    