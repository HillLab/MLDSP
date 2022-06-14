"""
@Daniel
Script for utilitaran functions
"""
from argparse import Action
from io import TextIOWrapper
from os import getenv
from pathlib import Path
from sys import stdout
from typing import Union

from MLDSP_core.__constants__ import methods_list


class Logger(object):
    """
    Class to handle logging operations
    """

    def __init__(self, path: Union[Path, str], filename: str):
        self.path = Path(path).resolve()
        self.filename = filename

    @property
    def path(self) -> Path:
        """
        getter of path property
        Returns:

        """
        return self.__path

    @path.setter
    def path(self, path) -> None:
        self.__path = Path(path).resolve()
        self.path.mkdir(exist_ok=True)

    def write(self, content) -> None:
        """
        Write content to log

        Args:
            content: Content to be written
        """
        with open(self.path.joinpath(self.filename), 'a') as logf:
            logf.write(content)


def uprint(*args, print_file: Union[str, Path, TextIOWrapper] = stdout):
    if isinstance(print_file, str) or isinstance(print_file, Path):
        parent = Path(print_file).parent
        parent.mkdir(exist_ok=True, parents=True)
        with open(print_file, 'a') as outfile:
            outfile.write('\n'.join(args))
    elif isinstance(print_file, TextIOWrapper):
        print(*args)
    else:
        raise TypeError


class PathAction(Action):
    """
    Class to set the action to store as path
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            parser.error("You need to provide a string with path or "
                         "filename")
        p = Path(values).resolve()
        if not (p.is_file() or p.is_dir()):
            if getenv(values) is not None:
                p = Path(getenv(values)).resolve()
            else:
                p.mkdir(parents=True)

        setattr(namespace, self.dest, p)


class HandleSpaces(Action):
    """
    Class to set the action to store as string
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            parser.error("You need to provide a string names or "
                         "filename")
        if isinstance(values, list):
            p = ' '.join(values)
        elif isinstance(values, str):
            p = values
        else:
            raise Exception('Wrong Type')
        choices = list(methods_list.keys())
        if p not in choices:
            exception_line = f"error: argument --method/-m: invalid " \
                             f"choice: '{p}' (choose from " \
                             f"{', '.join(choices)})"
            raise Exception(exception_line)
        setattr(namespace, self.dest, p)
