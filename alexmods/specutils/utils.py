#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utilities for dealing with spectra. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["calculate_fractional_overlap", "find_overlaps"]

import numpy as np
import time
import os, sys
from scipy import signal, ndimage
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, biweight_scale
from ..robust_polyfit import polyfit as rpolyfit

def calculate_fractional_overlap(interest_range, comparison_range):
    """
    Calculate how much of the range of interest overlaps with the comparison
    range.
    """

    if not (interest_range[-1] >= comparison_range[0] \
        and comparison_range[-1] >= interest_range[0]):
        return 0.0 # No overlap

    elif   (interest_range[0] >= comparison_range[0] \
        and interest_range[-1] <= comparison_range[-1]):
        return 1.0 # Total overlap 

    else:
        # Some overlap. Which side?
        if interest_range[0] < comparison_range[0]:
            # Left hand side
            width = interest_range[-1] - comparison_range[0]

        else:
            # Right hand side
            width = comparison_range[-1] - interest_range[0]
        return width/np.ptp(interest_range) # Fractional overlap


def find_overlaps(spectra, dispersion_range, return_indices=False):
    """
    Find spectra that overlap with the dispersion range given. Spectra are
    returned in order of how much they overlap with the dispersion range.

    :param spectra:
        A list of spectra.

    :param dispersion_range:
        A two-length tuple containing the start and end wavelength.

    :param return_indices: [optional]
        In addition to the overlapping spectra, return their corresponding
        indices.

    :returns:
        The spectra that overlap with the dispersion range provided, ranked by
        how much they overlap. Optionally, also return the indices of those
        spectra.
    """

    fractions = np.array([
        calculate_fractional_overlap(s.dispersion, dispersion_range) \
            for s in spectra])

    N = (fractions > 0).sum()
    indices = np.argsort(fractions)[::-1]
    overlaps = [spectra[index] for index in indices[:N]]

    """
    A faster, alternative method if sorting is not important:
    # http://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap/325964#325964    
    overlaps, indices = zip(*[(spectrum, i) \
        for i, spectrum in enumerate(spectra) \
            if  spectrum.dispersion[-1] >= dispersion_range[0] \
            and dispersion_range[-1] >= spectrum.dispersion[0]])
    """

    return overlaps if not return_indices else (overlaps, indices[:N])

def calculate_noise(y, window, method='madstd', verbose=True, full_output=False):
    """
    Calculate SNR a posteriori.
    Very inefficient but who cares!
    :param y:
        array of flux values to calculate SNR.
        Assumes equal or slowly changing x sampling
    :param window:
        pixel window to use
    :param method:
        madstd (default): use astropy.stats.mean_absolute_deviation
        betasigma: use PyAstronomy BSEqSamp
    :param verbose:
        if True (default), prints how long it takes to calculate the SNR.
    """
    assert method in ['madstd', 'betasigma']

    from scipy import signal
    ## Signal
    S = signal.medfilt(y, window)
    
    ## Noise Estimator
    if method == 'betasigma':
        from PyAstronomy import pyasl
        beq = pyasl.BSEqSamp()
        N = 1; j = 1
        def noise_estimator(x):
            return beq.betaSigma(x, N, j, returnMAD=True)[0]
    elif method == 'madstd':
        try:
            from astropy.stats import mean_absolute_deviation
        except ImportError:
            def noise_estimator(x):
                return np.nanmedian(np.abs(x-np.nanmedian(x)))
        else:
            noise_estimator = lambda x: mean_absolute_deviation(x, ignore_nan=True)
    
    ## Calculate noise
    noise = np.zeros(len(y))
    if verbose: start = time.time()
    # middle: calculate rolling value
    for i in range(window, len(noise)-window):
        noise[i] = noise_estimator(y[i-window:i+window])
    # edges: just use the same noise value
    noise[:window] = noise_estimator(y[:window])
    noise[-window:] = noise_estimator(y[-window:])
    
    if verbose:
        print("Noise estimate of {} points with window {} took {:.1f}s".format(len(noise), window, time.time()-start))

    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    # sigma = 1.4826 * MAD
    noise *= 1.4826
    
    if full_output:
        SNR = S/noise
        return noise, S, SNR
    return noise

def calculate_snr(*args, **kwargs):
    kwargs = kwargs.copy()
    kwargs["full_output"] = True
    noise, S, SNR = calculate_noise(*args, **kwargs)
    return SNR


def find_resolution(multispec_fname, initial_fwhm=.05, usepercentile=True, percentiles=[60, 80, 95], Rguess=None, full_output=False, useclip=True, findpeak=False, makeplot=True):
    """
    """
    from .spectrum import Spectrum1D
    from astropy.stats import sigma_clip, biweight
    from ..robust_polyfit import gaussfit
    from ..utils import find_distribution_peak
    import time
    arcs = Spectrum1D.read(multispec_fname, flux_ext=4)
    line_centers = np.loadtxt(os.path.dirname(__file__)+"/../data/linelists/thar_list", usecols=0)
    
    start = time.time()

    alllinefits = []
    allRmed = []
    allRerr = []
    allwmid = []
    allR = []
    for i, arc in enumerate(arcs):
        linefits = []
        wave = arc.dispersion
        flux = arc.flux
        wmin, wmax = wave[0], wave[-1]
        wmid = (wmin+wmax)/2.
        lcs = line_centers[(line_centers > wmin) & (line_centers < wmax)]
        for lc in lcs:
            fwhm = initial_fwhm
            # get subpiece of arc
            ii = (wave > lc - 5*fwhm) & (wave < lc + 5*fwhm)
            _x, _y = wave[ii], flux[ii]
            # guess amplitude, center, sigma
            p0 = [np.max(_y), lc, fwhm/2.355]
            try:
                popt = gaussfit(_x, _y, p0)
            except:
                pass
            else:
                if popt[0] > 0 and abs(popt[1]-lc) < .05:
                    linefits.append(popt)
        try:
            A, w, s = np.array(linefits).T
        except ValueError:
            print("This order did not have any good lines I guess")
            #allR.append(np.nan); allRmed.append(np.nan); allRerr.append(np.nan); allwmid.append(wmid)
            continue
        alllinefits.append(linefits)
        R = w/(s*2.355)
        if useclip: R = sigma_clip(R)
        if findpeak:
            if Rguess is None: Rguess = np.nanmedian(R)
            try:
                Rmed = find_distribution_peak(R, Rguess)
            except (ValueError, RuntimeError):
                print("--Could not find peak for arc {:02}".format(i))
                print("--{}".format(sys.exc_info()))
                Rmed = np.median(R)
        elif usepercentile:
            assert len(percentiles) == 3
            Rlo, Rmed, Rhi = np.percentile(R, percentiles)
            #Rerr = max(Rhi-Rmed, Rmed-Rlo)
            Rerr = (Rmed-Rlo, Rhi-Rmed)
        else:
            Rmed = biweight.biweight_location(R)
            Rerr = biweight.biweight_scale(R)
        allR.append(R); allRmed.append(Rmed); allRerr.append(Rerr); allwmid.append(wmid)
    if usepercentile:
        allRerr = np.array(allRerr).T
    
    if makeplot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.errorbar(allwmid, allRmed, yerr=allRerr, fmt='o')
        plt.show()

    if full_output:
        return allRmed, allRerr, allwmid, allR, arcs, alllinefits
    return allRmed, allRerr, allwmid




def find_peaks(flux,
               window = 51,niter = 5,
               clip_iter = 5,clip_sigma_upper = 5.0,clip_sigma_lower = 5.0,
               detection_sigma = 3.0,
               min_peak_dist_sigma = 5.0,
               gaussian_width = 1.0,
               make_fig=False):
    """
    * Subtract median filter (param "window")
    * Iterate: (param "niter")
        * Sigma clip, estimate noise (params clip_iter, clip_sigma_upper clip_sigma_lower)
        * Find peaks (param detection_sigma)
        * Remove peaks too close to previous (param min_peak_dist_sigma)
        * Fit Gaussians to peaks (initialize width at param gaussian_width)
    Returns:
        allpeakx: locations of peaks
        fullmodel: the model of all the gaussians
        If make_fig=True: fig, a plot showing all the peaks found at each iteration.
    """
    # This is the data we will try to fit with a
    # combination of Gaussians
    xarr = np.arange(len(flux))
    flux = flux - signal.medfilt(flux,window)
    continuum = models.Linear1D(slope=0, intercept=0)
    fullmodel = continuum
    
    allpeakx = []
    allpeaksigma = []
    
    fitter = fitting.LevMarLSQFitter()
    if make_fig: fig, axes = plt.subplots(niter)
    for iiter in range(niter):
        # Subtract existing peaks
        tflux = flux - fullmodel(xarr)
        # Estimate noise
        cflux = sigma_clip(tflux, 
                           iters=clip_iter,
                           sigma_upper=clip_sigma_upper,
                           sigma_lower=clip_sigma_lower)
        noise = np.std(cflux)
        # Find peaks in residual using gradient = 0
        # Only keep peaks above detection threshold
        deriv = np.gradient(tflux)
        peaklocs = (deriv[:-1] >= 0) & (deriv[1:] < 0) & \
            (tflux[:-1] > detection_sigma * noise)
        peakx = np.where(peaklocs)[0]
        peaky = flux[:-1][peaklocs]
        # Prune peaks too close to existing peaks
        peaks_to_keep = np.ones_like(peakx, dtype=bool)
        for ix,x in enumerate(peakx):
            z = (x-np.array(allpeakx))/np.array(allpeaksigma)
            if np.any(np.abs(z) < min_peak_dist_sigma):
                peaks_to_keep[ix] = False
        peakx = peakx[peaks_to_keep]
        peaky = peaky[peaks_to_keep]
        
        # Add new peaks to the model
        for x, y in zip(peakx, peaky):
            g = models.Gaussian1D(amplitude=y, mean=x,
                                  stddev=gaussian_width)
            fullmodel = fullmodel + g
        print("iter {}: {} peaks (found {}, added {})".format(
                iiter, fullmodel.n_submodels()-1,
                len(peaks_to_keep), len(peakx)))
        # Fit the full model
        fullmodel = fitter(fullmodel, xarr, flux, maxiter=200*(fullmodel.parameters.size+1))
        print(fitter.fit_info["message"], fitter.fit_info["ierr"])
        # Extract peak x and sigma
        peak_x_indices = np.where(["mean_" in param for param in fullmodel.param_names])[0]
        peak_y_indices = peak_x_indices - 1
        peak_sigma_indices = peak_x_indices + 1
        allpeakx = fullmodel.parameters[peak_x_indices]
        allpeaky = fullmodel.parameters[peak_y_indices]
        allpeaksigma = fullmodel.parameters[peak_sigma_indices]
        # Make a plot
        if make_fig:
            try:
                ax = axes[iiter]
            except:
                ax = axes
            ax.plot(xarr,flux)
            ax.plot(peakx,peaky,'ro')
            ax.plot(xarr,fullmodel(xarr), lw=1)
            ax.axhspan(-noise,+noise,color='k',alpha=.2)
            ax.plot(xarr,flux-fullmodel(xarr))
            ax.vlines(allpeakx, allpeaky*1.1, allpeaky*1.1+300, color='r', lw=1)
    if make_fig: return allpeakx, fullmodel, fig
    return allpeakx, fullmodel

## Quick and dirty signal processing
def replace_nans(x, value=0., msg=""):
    if np.any(~np.isfinite(x)):
        bad = ~np.isfinite(x)
        print("WARNING {}: {} bad pixels, replacing with {}".format(msg, np.sum(bad), value))
        x[bad] = value
    return x
def fast_smooth(flux, fwhm):
    flux = replace_nans(flux, msg="fast_smooth")
    sigma = fwhm / (2 * (2*np.log(2))**0.5)
    return ndimage.gaussian_filter1d(flux, sigma)
def fast_find_continuum(flux, kernel=51, Niter=3):
    cont = np.copy(flux)
    cont = replace_nans(cont, msg="fast_find_continuum")
    for i in range(Niter):
        cont = signal.medfilt(cont, kernel)
        kernel += 10
    return cont
def fast_find_continuum_polyfit(flux, kernel=51, Niter=3, degree=3, getcoeff=False):
    cont = fast_find_continuum(flux, kernel, Niter)
    cont = replace_nans(cont, msg="fast_find_continuum_polyfit")
    x = np.arange(len(flux))
    popt, sigma = rpolyfit(x, cont, degree)
    if getcoeff: return popt
    return np.polyval(popt, x)
def fast_find_noise(flux, **sigma_clip_kwargs):
    """ sigma clip then biweight scale """
    flux = replace_nans(flux, msg="fast_find_noise")
    clipped = sigma_clip(flux, **sigma_clip_kwargs)
    noise = biweight_scale(clipped)
    return noise
def fast_find_peaks(flux, cont=None,
                    noise=None, detection_sigma=2.0,
                    detection_threshold=None,
                    prune_dist=None):
    # Subtract continuum:
    if cont is not None: flux = flux.copy() - cont
    
    # Find threshold for peak finding
    if noise is None: noise = fast_find_noise(flux)
    if detection_threshold is None:
        detection_threshold = detection_sigma * noise
    
    # 1st derivative for peak finding
    dflux = np.gradient(flux)
    ii1 = flux > detection_threshold
    ii2 = dflux >= 0
    ii3 = np.zeros_like(ii2)
    ii3[:-1] = dflux[1:] < 0
    peaklocs = ii1 & ii2 & ii3
    peakx = np.where(peaklocs)[0]
    
    
    if prune_dist is not None:
        pass
    return peakx
