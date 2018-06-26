# coding: utf-8

""" Misc utility functions """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from six import string_types
import numpy as np
from .robust_polyfit import gaussfit

def struct2array(x):
    """ Convert numpy structured array of simple type to normal numpy array """
    Ncol = len(x.dtype)
    type = x.dtype[0].type
    assert np.all([x.dtype[i].type == type for i in range(Ncol)])
    return x.view(type).reshape((-1,Ncol))

def vac2air(lamvac):
    """
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Morton 2000
    """
    s2 = (1e4/lamvac)**2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return lamvac/n

def air2vac(lamair):
    """
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Piskunov
    """
    s2 = (1e4/lamair)**2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2)
    return lamair*n

def find_distribution_peak(x, x0, s0=None, bins='auto'):
    """
    Take a histogram of the data and find the location of the peak closest to x0
    x: data (no nan's allowed)
    x0: peak location guess
    s0: width of peak guess (default std(x))
    """
    h, x = np.histogram(x, bins=bins)
    x = (x[1:]+x[:-1])/2.
    # find positive peak locations based on derivative
    dh = np.gradient(h)
    peaklocs = np.where((dh[:-1] > 0) & (dh[1:] < 0))[0]
    if len(peaklocs)==0:
        raise ValueError("No peaks found!")
    # get the best peak
    xpeaks = x[peaklocs+1]
    bestix = np.argmin(np.abs(xpeaks - x0))
    xbest = x[bestix]
    ybest = h[bestix]
    if s0 is None:
        s_est = np.std(x)
    else:
        s_est = s0
    # fit a Gaussian and get the peak
    A, xfit, s = gaussfit(x, h, [ybest, xbest, s_est], maxfev=99999)
    return xfit

