import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter

from .specutils import Spectrum1D

def plot_spectrum(spec, wlmin=None, wlmax=None, ax=None,
                  dxmaj=None, dxmin=None, dymaj=None, dymin=None,
                  fillcolor="#cccccc",fillalpha=1,
                  **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    wave = spec.dispersion
    flux = spec.flux
    errs = spec.ivar**-0.5
    ii = np.ones(len(wave), dtype=bool)
    if wlmin is not None:
        ii = ii & (wave > wlmin)
    if wlmax is not None:
        ii = ii & (wave < wlmax)
    
    wave = wave[ii]
    flux = flux[ii]
    errs = errs[ii]
    y1 = flux-errs
    y2 = flux+errs

    fill_between_steps(ax, wave, y1, y2, alpha=fillalpha, facecolor=fillcolor, edgecolor=fillcolor)
    ax.plot(wave, flux, **kwargs)
    
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    if dxmaj is not None: ax.xaxis.set_major_locator(MultipleLocator(dxmaj))
    if dxmin is not None: ax.xaxis.set_minor_locator(MultipleLocator(dxmin))
    if dymaj is not None: ax.yaxis.set_major_locator(MultipleLocator(dymaj))
    if dymin is not None: ax.yaxis.set_minor_locator(MultipleLocator(dymin))
    return ax

def fill_between_steps(ax, x, y1, y2=0, h_align='mid', **kwargs):
    """
    Fill between for step plots in matplotlib.

    **kwargs will be passed to the matplotlib fill_between() function.
    """

    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)
