#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Spectroscopy-related utilities. """

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))
logger.addHandler(handler)

from . import motions
from .spectrum import Spectrum1D
from .rv import *
from .utils import *
try:
    from .continuum import fit_continuum_lsq
except:
    print("Could not import continuum GUI (you are probably in python 2)")
