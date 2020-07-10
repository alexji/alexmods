#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A standardized interface to the MOOG(SILENT) radiative transfer code. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import six

__all__ = ["abundance_cog", "synthesize", "blends_cog", "RTError"]

if six.PY2:
    # See stackoverflow.com/questions/19913653/no-unicode-in-all-for-a-packages-init
    __all__ = [_.encode("ascii") for _ in __all__]

from .cog import abundance_cog
from .blends import blends_cog
from .synthesis import synthesize
from .utils import RTError
