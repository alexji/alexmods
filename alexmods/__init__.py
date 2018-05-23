#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import logging
import os

# Software version.
__version__ = "0.1"

# Python 2/3 compatibility:
safe_check_output = lambda x, shell=True: check_output(x, shell=shell).decode(
    "ascii", "ignore")

# Set up logging.
logger = logging.getLogger("alexmods")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))

logger.addHandler(handler)

# Import base level things
from .helloworld import hello

