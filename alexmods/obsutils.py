#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

"""
Utilities for observation planning
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import logging

from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.table import Table
from astropy import units as u
from astropy.time import Time

from imp import reload
from alexmods import utils
box_select = utils.box_select

__all__ = ["mike_snr_calculator","mike_texp_calculator"]

def mike_snr_calculator(bp,rp,texp,blue0=19.4,red0=18.5,slitloss=0.7):
    # these are in AB mags, but let's just go with it
    A_per_pix_B = 0.02
    count_rate_B = A_per_pix_B * 10**(-0.4*(bp-blue0)) * slitloss
    snr_B = np.sqrt(count_rate_B*texp)
    
    A_per_pix_R = 0.05
    count_rate_R = A_per_pix_R * 10**(-0.4*(rp-red0)) * slitloss
    snr_R = np.sqrt(count_rate_R*texp)
    
    return snr_B, snr_R
def mike_texp_calculator(bp, rp, snr, blue0=19.4,red0=18.5,slitloss=0.7):
    A_per_pix_B = 0.02
    count_rate_B = A_per_pix_B * 10**(-0.4*(bp-blue0)) * slitloss
    texp_B = snr**2 / count_rate_B
    
    A_per_pix_R = 0.05
    count_rate_R = A_per_pix_R * 10**(-0.4*(rp-red0)) * slitloss
    texp_R = snr**2 / count_rate_R
    
    return texp_B, texp_R

def add_month_lines(ax, text_y=0, radeg=False):
    ylim = ax.get_ylim()
    month_ra = [7,9,11,13,15,17,19,21,23,1,3,5]
    if radeg: month_ra = [x*360/24 for x in month_ra]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.vlines(month_ra,ylim[0],ylim[1],alpha=.3)
    for x,month in zip(month_ra, months):
        ax.text(x+.5,text_y,month,rotation='vertical',ha='left',va='center')
    if radeg:
        ax.xaxis.set_major_locator(plt.MultipleLocator(60))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(15))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
