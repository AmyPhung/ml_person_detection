"""Module of Actions subclassing argparse.Action for use in cli tools.

These classes perform various checks and conversions on command line
arguments. This is to make these checks take up less space in the main
modules and standardize them for use across tools.

For documentation on subclassing argparse.Actions, see:
    https://docs.python.org/3/library/argparse.html#action-classes

Examples:
    cli.add_argument('file', action=VerifyPathfileAction)

"""
from argparse import Action
from pathlib import Path

class VerifyPathDirAction(Action):
    """Verify directory arg exists."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Convert value to pathlib.Path, verify that it is dir."""
        path = Path(values).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"dir: {path}")

        setattr(namespace, self.dest, path)


class VerifyPathFileAction(Action):
    """Verify file arg exists."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Convert value to pathlib.Path, verify that it is folder."""
        path = Path(values).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"'{path}' does not exist as file")

        setattr(namespace, self.dest, path)


class CheckPathFileExistsAction(Action):
    """Check if file arg exists."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Convert value to pathlib.Path, write is_file to csv_exists."""
        path = Path(values).resolve()
        setattr(namespace, self.dest, path)
        setattr(namespace, 'csv_exists', path.is_file())
