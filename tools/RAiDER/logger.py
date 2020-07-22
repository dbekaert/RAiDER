#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Rohan Weeden
# Copyright 2020, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Global logging configuration
"""
import logging
import os
import sys
from logging import FileHandler, Formatter, StreamHandler


# Inspired by
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class UnixColorFormatter(Formatter):
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLORS = {
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self, fmt=None, datefmt=None, style="%", use_color=True):
        super().__init__(fmt, datefmt, style)
        if use_color:
            self.formatMessage = self.formatMessageColor

    def formatMessageColor(self, record):
        message = super().formatMessage(record)
        color = self.COLORS.get(record.levelno)
        if color:
            return "".join([color, message, self.reset])
        return message


logger = logging.getLogger("RAiDER")
logger.setLevel(logging.INFO)

stdout_handler = StreamHandler(sys.stdout)
stdout_handler.setFormatter(UnixColorFormatter(use_color=os.name != "nt"))

errorfile_handler = FileHandler("error.log")
errorfile_handler.setFormatter(Formatter(
    "[{asctime}] {funcName:>20}:{lineno:<5} {levelname:<10} {message}",
    style="{"
))
errorfile_handler.setLevel(logging.WARNING)

logger.addHandler(stdout_handler)
logger.addHandler(errorfile_handler)
