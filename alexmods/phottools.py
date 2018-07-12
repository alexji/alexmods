# coding: utf-8
""" Some photometry tools for stellar spectroscopists """
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from scipy import interpolate

import logging
import os, sys, time
logger = logging.getLogger(__name__)

__all__ = []

### Setup Jordi06
def get_jordi06_coeffs(type):
    if type==0: # Combined Pop I/Pop II
        a_Bmg = 0.313; e_a_Bmg = 0.003
        b_Bmg = 0.219; e_b_Bmg = 0.002
        a_Vmg =-0.565; e_a_Vmg = 0.001
        b_Vmg =-0.016; e_b_Vmg = 0.001
    elif type==1: # Pop I
        a_Bmg = 0.312; e_a_Bmg = 0.003
        b_Bmg = 0.219; e_b_Bmg = 0.002
        a_Vmg =-0.573; e_a_Vmg = 0.002
        b_Vmg =-0.016; e_b_Vmg = 0.002
    elif type==2: # Pop II
        a_Bmg = 0.349; e_a_Bmg = 0.009
        b_Bmg = 0.245; e_b_Bmg = 0.006
        a_Vmg =-0.569; e_a_Vmg = 0.007
        b_Vmg = 0.021; e_b_Vmg = 0.004
    else:
        raise ValueError("Type must be 0, 1, 2 (got {})".format(type))
    return a_Bmg, b_Bmg, a_Vmg, b_Vmg, e_a_Bmg, e_b_Bmg, e_a_Vmg, e_b_Vmg
    
def _gmr_to_BmV(gmr,geterr=True,type=0):
    a_Bmg, b_Bmg, a_Vmg, b_Vmg, e_a_Bmg, e_b_Bmg, e_a_Vmg, e_b_Vmg = get_jordi06_coeffs(type)
    # Calculate middle
    Bmg = a_Bmg*gmr + b_Bmg
    Vmg = a_Vmg*gmr + b_Vmg
    BmV = Bmg - Vmg

    if not geterr: return BmV

    # Calculate 1 sigma error estimate
    if gmr >= 0:
        Bmg_max = (a_Bmg+e_a_Bmg)*gmr+(b_Bmg+e_b_Bmg)
        Bmg_min = (a_Bmg-e_a_Bmg)*gmr+(b_Bmg-e_b_Bmg)
        Vmg_max = (a_Vmg+e_a_Vmg)*gmr+(b_Vmg+e_b_Vmg)
        Vmg_min = (a_Vmg-e_a_Vmg)*gmr+(b_Vmg-e_b_Vmg)
    else:
        Bmg_max = (a_Bmg-e_a_Bmg)*gmr+(b_Bmg+e_b_Bmg)
        Bmg_min = (a_Bmg+e_a_Bmg)*gmr+(b_Bmg-e_b_Bmg)
        Vmg_max = (a_Vmg-e_a_Vmg)*gmr+(b_Vmg+e_b_Vmg)
        Vmg_min = (a_Vmg+e_a_Vmg)*gmr+(b_Vmg-e_b_Vmg)
        
    BmV_max = Bmg_max-Vmg_min
    BmV_min = Bmg_min-Vmg_max
    return BmV_min,BmV,BmV_max
jordi06_gmr_to_BmV = np.vectorize(_gmr_to_BmV)

if __name__=="__main__":
    gmag, rmag = np.nan, np.nan
    
    # Pic2
    gmag0, rmag0 = 17.39, 16.69
    BmVmin, BmV, BmVmax = jordi06_gmr_to_BmV(gmag0-rmag0,type=2)
    print(BmVmin, BmV, BmVmax)
    
    
    """
    Teff:
    Two colors (+ errors)
    Dereddening (+ errors)
    Conversion to filters for color-Teff relations (+ errors)
    --> Teff + errors
    """
    
    """
    logg with distance
    One color (+ errors)
    Mass assumption
    Teff (+ error)
    Distance (+ error)
    Bolometric correction (+ error): needs logg, Teff input
    """
    
    """
    logg with isochrone
    Need Teff + error
    Need isochrone
    """
    
    """
    logg, Teff from fitting to Casagrande + Vandenberg grid
    """
