"""
@Daniel
Script for utilitaran functions
"""
from pathlib import Path
from typing import Union


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
