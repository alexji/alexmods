from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from six import string_types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from . import read_data as rd
from .smhutils import element_to_atomic_number

def prepare_data(elems, XH=None, logeps=None, XFe=None, errs=None,
                 drop_Z=[24]):
    """
    Convert data to pd.Series(logeps, index=Z)
    If errs < 0, it is assumed to be upper limit.
    Return pd.Series(epsvals, index=Z), pd.Series(epserr, index=Z), pd.Series(epslim, index=Z)
    
    drop_Z = [24] by default.
    This is a list of elements to remove from the data (not yet implemented)
    """
    # Check for errors
    assert errs is not None, "Must specify errors for chi2!"
    errs = np.array(errs)
    # Check for one of XH, logeps, XFe
    checkinput = [XH is not None, logeps is not None, XFe is not None]
    assert np.sum(checkinput) == 1, np.where(checkinput)[0]
    if XFe is not None: raise NotImplementedError("Specify XH or logeps for now")
    # Check array lengths
    assert len(elems) == len(errs)
    if XH is not None:
        assert len(XH)==len(elems)
        XH = np.array(XH)
    if logeps is not None:
        assert len(logeps)==len(elems)
        logeps = np.array(logeps)
    if XFe is not None:
        assert len(XFe)==len(elems)
        XFe = np.array(XFe)
    # Check elems format, convert to Z
    if np.all([isinstance(elem, (int, np.integer, float, np.float)) for elem in elems]):
        elems = np.array(elems).astype(int)
    elif np.all([isinstance(elem, string_types) for elem in elems]):
        print("Elements are strings, converting to Z (this is untested)")
        elems = np.array([element_to_atomic_number(elem) for elem in elems]).astype(int)
    else:
        raise ValueError("Elems must be all numeric or all strings: type(elems)={}, {}".format(type(elems),elems))
    # TODO convert XFe to XH
    if XFe is not None:
        iiFe = elems==26
        assert np.sum(iiFe) == 1, (elems,iiFe)
        FeH = XFe[iiFe][0]
        print("Assuming [Fe/H]={:.2f}".format(FeH))
        XH = XFe.copy()
        XH[~iiFe] = XFe[~iiFe]+FeH
    # Convert XH to logeps
    if XH is not None:
        XH = pd.Series(XH, index=elems)
        solar = rd.get_solar(elems)
        logeps = XH + solar
    else:
        logeps = pd.Series(logeps, index=elems)
    epserr = pd.Series(errs, index=elems)
    iilim = epserr < 0
    epsval = logeps[~iilim].astype(float)
    epslim = logeps[iilim].astype(float)
    epserr = epserr[~iilim].astype(float)
    return epsval, epslim, epserr

def load_hw10_starfit():
    """
    Set up data for fitting
    """
    hw10 = rd.load_hw10(as_number=True)
    columns_to_use = np.array([isinstance(col, (int, np.integer)) for col in hw10.columns])
    hw10logN = np.log10(hw10[hw10.columns[columns_to_use]])
    return hw10, hw10logN

def find_best_models(hw10, hw10logN, epsval, epserr, epslim, name=""):
    """
    Return the best-fitting HW10 models.
    * Find best offsets to minimize chi2 for epsval and epserr
    * Reject all models with any abundances > epslim
    
    Returns:
    best_models (sorted by chi2), all chi2 (order of hw10logN), valid indices, model offsets
    
    TODO: Check the ones with chi2 < the best chi2 and see if shifting those will do better
    
    Note: solving for the offset is done analytically.
    For model $k$, data $x_i$, errors $\sigma_i$, and offset $\delta$

    $$ \chi^2(k) = \sum_i \left(\frac{x_i - \mu_{k,i} + \delta}{\sigma_i}\right)^2 $$

    Solving for $\partial \chi^2/\partial \delta = 0$, we get

    $$ \delta \sum_i \frac{1}{\sigma_i^2} = -\sum_i \frac{x_i - \mu_{k,i}}{\sigma_i^2} $$

    """
    # Get data ready
    assert np.all(epsval.index == epserr.index)
    modelvals = hw10logN[epsval.index].values
    modellims = hw10logN[epslim.index].values
    x = epsval.values
    e = epserr.values
    e2= e*e
    ul= epslim.values
    wtot = np.sum(1./e2)
    
    # Solve for bestfit offsets to model
    # delta * wtot = - sum((x-mu)/sigma^2)
    offsets = -np.sum((x - modelvals)/e2, axis=1)/wtot
    new_modelvals = modelvals - offsets[:,np.newaxis]
    new_modellims = modellims - offsets[:,np.newaxis]
    
    # Reject things that do not satisfy the upper limits
    # I believe this differs from the starfit algorithm,
    #    since there it will adjust the offsets until 
    valid_ul = np.all(new_modellims < ul, axis=1)
    
    chi2 = np.sum(((x-new_modelvals)/e)**2,axis=1)
    
    iisort = np.argsort(chi2[valid_ul])
    Nelem = len(hw10logN.columns)
    best_models = hw10[valid_ul].iloc[iisort] #- np.tile(np.array(offsets[valid_ul,np.newaxis][iisort]), Nelem)
    best_models["chi2"] = chi2[valid_ul][iisort]
    # Solving logeps(X) - 12 = log(NX)/log(NH) -->
    # log MH = 12 + logN(X) - logeps(X), where N(X) = M(X)/A(X) (Msun/amu) is computed in load_hw10, assuming A(H)=1.0
    best_models["Dilution"] = 12 + offsets[valid_ul][iisort]
    
    best_models.index = np.arange(len(best_models))+1
    best_models.columns.name = name
    best_logN = hw10logN[valid_ul].iloc[iisort] - np.tile(np.array(offsets[valid_ul,np.newaxis][iisort]), Nelem)
    
    return best_models, best_logN, chi2, valid_ul, offsets


def plot_chi2_all(hw10, valid_ul, chi2, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1,3,figsize=(18,6))
    else:
        assert len(axes)==3
        fig = axes[0].figure
    for col, ax in zip(["Mass","Energy","Mixing"], axes.flat):
        ax.plot(hw10[col],chi2,'k,')
        ax.plot(hw10[col][valid_ul],chi2[valid_ul],'r.')
        ax.set_xlabel(col)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_yscale('log')
    return fig
def plot_abund_fit(epsval, epserr, epslim, logNmodel, offset=0.,
                   fmt="s", color="r", ms=6,
                   model_only=False,
                   mcolor = 'k', mlw=1, mls='-', mlabel=None, mkws={},
                   ax = None, plot_XH=False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    Zii = np.array([isinstance(x, int) and x > 5 for x in logNmodel.index])
    if plot_XH:
        solar = rd.get_solar(logNmodel.index[Zii])
        ax.plot(logNmodel.index[Zii], logNmodel[Zii]-solar, color=mcolor, lw=mlw, ls=mls, label=mlabel, **mkws)
        if not model_only:
            ax.errorbar(epsval.index, epsval.values-solar.loc[epsval.index], yerr=epserr.values, fmt=fmt, color=color, ecolor=color, ms=ms, label=None)
            ax.errorbar(epslim.index, epslim.values-solar.loc[epslim.index], yerr=.1, fmt='none', color=color, ecolor=color, uplims=True, label=None)
    else:
        ax.plot(logNmodel.index[Zii], logNmodel[Zii], color=mcolor, lw=mlw, ls=mls, label=mlabel, **mkws)
        if not model_only:
            ax.errorbar(epsval.index, epsval.values, yerr=epserr.values, fmt=fmt, color=color, ecolor=color, ms=ms)
            ax.errorbar(epslim.index, epslim.values, yerr=.1, fmt='none', color=color, ecolor=color, uplims=True)
    
    
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(5,30)
    ax.set_xlabel("Z")
    if plot_XH:
        XHval = epsval.values - solar.loc[epsval.index]
        XHlim = epslim.values - solar.loc[epslim.index]
        minXH = min(np.min(XHval), np.min(XHlim))
        maxXH = max(np.max(XHval), np.max(XHlim))
        minXH = np.floor(minXH)
        maxXH = np.ceil(maxXH)
        ax.set_ylim(minXH, maxXH)
        ax.set_ylabel(r"[X/H]")
    else:
        mineps = np.floor(min(np.min(epsval-epserr), np.min(epslim))) - 1
        maxeps = np.ceil(max(np.max(epsval+epserr), np.max(epserr))) + 1
        ax.set_ylim(mineps, maxeps)
        ax.set_ylabel(r"$\log\epsilon(X)$")
    return fig
def generate_labels(best_models, Mdil=False):
    if Mdil:
        labels = [r"M={:.1f}$M_\odot$ E={:.1f}B $\xi$={:.2e} $\log M_{{\rm dil}}$={:.1f} $\chi^2$={:.1f}".format(
                *tuple(model[["Mass","Energy","Mixing","Dilution","chi2"]])) \
                      for i,model in best_models.iterrows()]
    else:
        labels = [r"M={:.0f}$M_\odot$ E={:.1f}B $\xi$={:.2e} $\chi^2$={:.1f}".format(
                *tuple(model[["Mass","Energy","Mixing","chi2"]])) \
                      for i,model in best_models.iterrows()]
    return labels
    
def plot_many_abund_fits(nrow, ncol,
                         epsval, epserr, epslim,
                         logNmodels, labels,
                         subplotwidth=8,subplotheight=6,
                         **kwargs):
    assert nrow*ncol == len(logNmodels)
    assert len(logNmodels)==len(labels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*subplotwidth,nrow*subplotheight))
    for irow in range(nrow):
        for icol in range(ncol):
            imodel = irow*ncol + icol
            model = logNmodels.iloc[imodel]
            label = labels[imodel]
            plot_abund_fit(epsval, epserr, epslim, model,
                           ax = axes[irow,icol], mlabel=label,
                           **kwargs)
            axes[irow,icol].legend(loc='upper right')
    return fig

def plot_model_parameters(best_models, best_logN, 
                          scattermin=1, scattermax=250,
                          sigma=2.0, minlogMdil=2.0, **kwargs):
    chi2 = best_models["chi2"].values
    minchi2 = np.min(chi2)
    maxchi2 = minchi2 + stats.chi2.ppf(stats.norm.cdf(sigma)-stats.norm.cdf(-sigma),4) # 4 params
    chi2sf = stats.chi2.sf(chi2-minchi2, 4)
    scattersize = scattermin + (scattermax-scattermin)*chi2sf
    
    valid_for_plot = np.logical_and(chi2 < maxchi2, best_models["Dilution"] > minlogMdil)
    cols = ["Mass","Energy","Mixing","Dilution"]
    limits = [[np.min(best_models[col]), np.max(best_models[col])] for col in cols]
    limits[cols.index("Dilution")][0] = minlogMdil
    limits[cols.index("Energy")][0] = 0
    limits[cols.index("Mass")][0] = 0
    
    colbins = [np.arange(10,100,  10),
               np.arange(0, 10.1, .5),
               np.arange(0,0.252,.02),
               np.arange(2,7,.5)]
    fig, axes = plt.subplots(4,4,figsize=(4*6,4*6))
    for i1, col1 in enumerate(cols):
        for i2, col2 in enumerate(cols):
            ax = axes[i1,i2]
            if i1==i2:
                x = np.array(best_models[col1][valid_for_plot])
                ax.hist(x[np.isfinite(x)],weights=chi2sf[valid_for_plot][np.isfinite(x)],
                        bins=colbins[i1])
                ax.set_xlabel(col1)
                ax.set_xlim(limits[i1])
            else:
                ax.scatter(best_models[col1][valid_for_plot], best_models[col2][valid_for_plot], s=scattersize[valid_for_plot], **kwargs)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_xlim(limits[i1])
                ax.set_ylim(limits[i2])
    return fig
