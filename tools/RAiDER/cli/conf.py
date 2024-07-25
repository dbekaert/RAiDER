from pathlib import Path
from typing import Optional


LOGGER_PATH: Optional[Path] = None


def setLoggerPath(path: Optional[Path]) -> None:
    global LOGGER_PATH
    LOGGER_PATH = path
