from typing import Optional
from pathlib import Path

LOGGER_PATH: Optional[Path] = None


def setLoggerPath(path: Path) -> None:
    global LOGGER_PATH
    LOGGER_PATH = path
