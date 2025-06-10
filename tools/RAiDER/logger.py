# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Rohan Weeden
# Copyright 2020, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Global logging configuration."""

import logging
import os
import sys
from logging import FileHandler, Formatter, LogRecord, StreamHandler
from pathlib import Path

import RAiDER.cli.conf as conf


# Inspired by
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class UnixColorFormatter(Formatter):
    yellow = '\x1b[33;21m'
    red = '\x1b[31;21m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    COLORS = {
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self, fmt: str = None, datefmt: str = None, style: str = '%', use_color: bool=True) -> None:
        super().__init__(fmt, datefmt, style)
        # Save the old function so we can call it later
        self.__formatMessage = self.formatMessage
        if use_color:
            self.formatMessage = self.formatMessageColor

    def formatMessageColor(self, record: LogRecord) -> str:
        message = self.__formatMessage(record)
        color = self.COLORS.get(record.levelno)
        if color:
            message = ''.join([color, message, self.reset])
        return message


class CustomFormatter(UnixColorFormatter):
    """Adds levelname prefixes to the message on warning or above."""
    def formatMessage(self, record: LogRecord) -> str:
        message = super().formatMessage(record)
        if record.levelno >= logging.WARNING:
            message = ': '.join((record.levelname, message))
        return message


#####################################
# DEFINE THE LOGGER
if conf.LOGGER_PATH is not None:
    logger_path = conf.LOGGER_PATH
else:
    logger_path = Path.cwd()

logger = logging.getLogger('RAiDER')
logger.setLevel(logging.DEBUG)

stdout_handler = StreamHandler(sys.stdout)
stdout_handler.setFormatter(CustomFormatter(use_color=os.name != 'nt'))
stdout_handler.setLevel(logging.DEBUG)

debugfile_handler = FileHandler(logger_path / 'debug.log')
debugfile_handler.setFormatter(
    Formatter('[{asctime}] {levelname:<10} {module} {exc_info} {funcName:>20}:{lineno:<5} {message}', style='{')
)
debugfile_handler.setLevel(logging.DEBUG)

errorfile_handler = FileHandler(logger_path / 'error.log')
errorfile_handler.setFormatter(
    Formatter('[{asctime}] {levelname:<10} {module:<10} {exc_info} {funcName:>20}:{lineno:<5} {message}', style='{')
)
# <time-tag>, <log level>, <workflow>, <module>, <error code>, <error location>, <logged message>
errorfile_handler.setLevel(logging.WARNING)

logger.addHandler(stdout_handler)
logger.addHandler(errorfile_handler)
logger.addHandler(debugfile_handler)
