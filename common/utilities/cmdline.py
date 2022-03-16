import argparse
import os.path


def CheckExt(choices):
    """Wrapper to return the class
    """
    class Act(argparse.Action):
        """Class to allow checking of filename extensions in argparse. Taken
        from https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
        """
        def __call__(self, parser, namespace, fname, option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(
                    option_string) if option_string else ''
                parser.error("Wrong filetype: file doesn't end with {}{}".format(
                    choices, option_string))
            else:
                setattr(namespace, self.dest, fname)

    return Act
