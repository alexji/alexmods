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
from scipy.stats import norm as norm_distr
from scipy.stats import pearsonr
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, biweight_scale, median_absolute_deviation
from ..robust_polyfit import polyfit as rpolyfit
from .spectrum import Spectrum1D, common_dispersion_map2
from collections import OrderedDict

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
    print("fast_find_noise: sigma-clipped {}/{}".format(np.sum(~clipped.mask),len(clipped)))
    noise = biweight_scale(clipped[~clipped.mask])
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

def cosmic_ray_reject(flux, ivar, sigma=10., minflux=-100, use_mad=False, verbose=False):
    """
    flux (n_frame, n_pix)
    """
    assert flux.shape==ivar.shape
    TINYNUMBER=1e-8

    n_frame, n_pix = flux.shape
    
    scale_factor = np.nanmedian(flux, axis=1)
    scaled_flux = (flux.T/scale_factor).T
    scaled_median_flux = np.median(scaled_flux, axis=0)
    
    errs = ivar**-0.5
    if use_mad:
        scaled_errs = 1.4826 * median_absolute_deviation(scaled_flux, axis=0)
        for iframe in range(n_frame):
            errs[iframe] = scaled_errs * scale_factor[iframe]

    new_flux = flux.copy()
    new_ivar = ivar.copy()
    new_mask = np.zeros_like(new_flux, dtype=bool)
    for iframe in range(n_frame):
        medianspec = scale_factor[iframe] * scaled_median_flux
        mask1 = np.abs(flux[iframe]) > sigma * errs[iframe] + medianspec
        mask2 = flux[iframe] < minflux
        mask3 = ~(np.isfinite(flux[iframe]) & np.isfinite(ivar[iframe]))
        mask = mask1 | mask2 | mask3
        new_flux[iframe,mask] = medianspec[mask]
        new_ivar[iframe,mask] = TINYNUMBER
        new_mask[iframe,mask] = True
        if verbose:
            print("frame {}: {}px high, {}px low, {}px nan, {} total".format(
                iframe, mask1.sum(), mask2.sum(), mask3.sum(), mask.sum()))
    return new_flux, new_ivar, new_mask

def find_lines(wave, flux, ivar):
    """
    Automatically find all locations of lines in a spectrum
    """
    raise NotImplementedError

def rescale_snr(specwave, flux=None, ivar=None,
                x1=None, x2=None,
                cont=None, cont_kernel=51, cont_Niter=3,
                make_fig=False,
                **sigma_clip_kwargs):
    """
    Take biweight standard deviation of x_i/sigma_i of sigma-clipped data,
    rescale ivar so that standard deviation is 1
    Returns a Spectrum1D object
    """
    if flux is None and ivar is None:
        assert isinstance(specwave, Spectrum1D)
        spec = specwave
        wave, flux, ivar = spec.dispersion, spec.flux, spec.ivar
        meta = spec.metadata
    else:
        wave = specwave
        assert len(wave) == len(flux)
        assert len(wave) == len(ivar)
        meta = OrderedDict({})
    if cont is None:
        cont = fast_find_continuum(flux, cont_kernel, cont_Niter)
    else:
        assert len(cont)==len(flux)
    errs = ivar**-0.5
    errs[errs > 10*flux] = np.nan
    
    iirescale = np.ones_like(wave,dtype=bool)
    if x1 is not None: iirescale = iirescale & (wave > x1)
    if x2 is not None: iirescale = iirescale & (wave < x2)
    
    norm = flux[iirescale]/cont[iirescale]
    normerrs = errs/cont[iirescale]
    
    z = (norm-1.)/normerrs
    clipped = sigma_clip(z[np.isfinite(z)], **sigma_clip_kwargs)
    noise = biweight_scale(clipped[~clipped.mask])
    print("Noise is {:.2f} compared to 1.0".format(noise))
    
    new_ivar = ivar / (noise**2.)
    
    outspec = Spectrum1D(wave, flux, new_ivar, meta)
    if make_fig:
        newz = z/noise
        newerrs = errs*noise
        newnormerrs = normerrs*noise
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,3,figsize=(12,6))
        ax = axes[0,0]
        ax.plot(wave, flux)
        ax.plot(wave, errs)
        ax.plot(wave, newerrs)
        ax.plot(wave, cont, color='k', ls=':')
        ax.set_xlabel('wavelength'); ax.set_ylabel('counts')
        ax = axes[1,0]
        ax.plot(wave, norm)
        ax.plot(wave, normerrs)
        ax.plot(wave, newnormerrs)
        ax.axhline(1,color='k',ls=':')
        ax.set_xlabel('wavelength'); ax.set_ylabel('norm'); ax.set_ylim(0,1.2)
        ax = axes[1,1]
        ax.plot(wave, z)
        ax.plot([np.nan],[np.nan]) # hack to get the right color
        ax.plot(wave, newz)
        ax.axhline(0,color='k',ls=':')
        ax.set_xlabel('wavelength'); ax.set_ylabel('z'); ax.set_ylim(-7,7)
        
        ax = axes[0,1]
        bins = np.linspace(-7,7,100)
        binsize = np.diff(bins)[1]
        ax.plot(bins, norm_distr.pdf(bins)*np.sum(np.isfinite(z))*binsize, color='k')
        ax.hist(z[np.isfinite(z)], bins=bins)
        ax.hist(clipped[~clipped.mask], bins=bins, histtype='step')
        ax.hist(newz[np.isfinite(newz)], bins=bins, histtype='step')
        ax.set_xlabel('z'); ax.set_xlim(-7,7)
        
        ax = axes[0,2]
        zfinite = z.copy()
        zfinite[~np.isfinite(zfinite)] = 0.
        autocorr = np.correlate(zfinite, zfinite, mode="same")
        ax.plot(np.arange(len(flux)), autocorr, '.-')
        ax.axvline(len(flux)//2)
        ax.set_xlim(len(flux)//2-10,len(flux)//2+10,)
        ax.set_xlabel("pixel"); ax.set_ylabel("autocorrelation(z)")
        
        
        z1,z2 = -10,10
        zarr1 = np.zeros((len(z)-1,2))
        zarr1[:,0] = z[:-1]
        zarr1[:,1] = z[1:]
        zarr1 = zarr1[np.sum(np.isfinite(zarr1),axis=1)==2]
        zarr2 = np.zeros((len(z)-2,2))
        zarr2[:,0] = z[:-2]
        zarr2[:,1] = z[2:]
        zarr2 = zarr2[np.sum(np.isfinite(zarr2),axis=1)==2]
        #ax = axes[0,2]
        #ax.plot([z1,z2],[z1,z2],'k:')
        #ax.plot(z[:-1], z[1:], '.', alpha=.3)
        #ax.set_title("r={:+.2}".format(pearsonr(zarr1[:,0],zarr1[:,1])[0]))
        #ax.set_xlabel("z(pixel)"); ax.set_ylabel("z(pixel+1)")
        #ax.set_xlim(z1,z2); ax.set_ylim(z1,z2)

        ax = axes[1,2]
        ax.plot([z1,z2],[z1,z2],'k:')
        ax.plot(z[:-2], z[2:], '.', alpha=.3)
        ax.set_title("r={:+.2}".format(pearsonr(zarr2[:,0],zarr2[:,1])[0]))
        ax.set_xlabel("z(pixel)"); ax.set_ylabel("z(pixel+2)")
        ax.set_xlim(z1,z2); ax.set_ylim(z1,z2)
        
        fig.tight_layout()
        
        return fig, outspec, noise
    
    return outspec, noise

def weighted_coadd(spectra, common_dispersion=None, mask_cosmics=False,
                   sigma=5, TINY_NUMBER=1e-12, verbose=True):
    ## Interpolate spectra onto a common dispersion
    if common_dispersion is None:
        common_dispersion = common_dispersion_map2(spectra)
    Nspecs = len(spectra)
    Npix = len(common_dispersion)
    fluxs = np.zeros((Nspecs, Npix))
    ivars = np.zeros((Nspecs, Npix))
    for i, spec in enumerate(spectra):
        newspec = spec.linterpolate(common_dispersion)
        fluxs[i] = newspec.flux
        ivars[i] = newspec.ivar
    ## Remove bad values
    fluxs[np.isnan(fluxs)] = 0.
    ivars[np.isnan(ivars) | (ivars < TINY_NUMBER)] = TINY_NUMBER
    ## Normalize to a common scale so we can take a weighted mean
    total_flux = np.median(fluxs, axis=1)
    normfluxs = fluxs/total_flux[:,np.newaxis]
    normivars = ivars*total_flux[:,np.newaxis]**2
    normerrss = normivars**-0.5
    ## Remove outlier pixels if desired
    if mask_cosmics:
        orig_masked = normivars <= TINY_NUMBER
        mednormflux = np.median(normfluxs, axis=0)
        mask = np.abs(normfluxs - mednormflux)/normerrss > sigma
        normfluxs[mask] = 0.
        normivars[mask] = 0.
        if verbose:
            print("Masked {} pixels (added {})".format(mask.sum(), mask.sum()-orig_masked.sum()))
    ## Perform weighted coadd
    num_nonzero = np.sum(normivars > TINY_NUMBER, axis=0)
    avgflux = np.sum(normfluxs*normivars,axis=0)/np.sum(normivars,axis=0)
    #avgerrs = np.sqrt(np.sum((normfluxs-avgflux[np.newaxis,:])**2*normivars,axis=0)/np.sum(normivars,axis=0))
    #avgerrs = avgerrs/np.sqrt(num_nonzero-1)
    #avgivar = avgerrs**-2.
    #avgivar[~np.isfinite(avgivar)] = TINY_NUMBER
    avgivar = np.sum(normivars, axis=0)
    ## Renormalize to original units
    totnorm = np.sum(total_flux)
    finalflux = totnorm * avgflux
    finalivar = avgivar / totnorm**2
    
    return Spectrum1D(common_dispersion, finalflux, finalivar, spectra[0].metadata)
    

def make_multispec(outfname, bands, bandids, header=None):
    assert len(bands) == len(bandids)
    Nbands = len(bands)
    # create output image array
    # Note we have to reverse the order of axes in a fits file
    shape = bands[0].shape
    output = np.zeros((Nbands, shape[1], shape[0]))
    for k, (band, bandid) in enumerate(zip(bands, bandids)):
        output[k] = band.T
    if Nbands == 1:
        output = output[0]
        klist = [1,2]
        wcsdim = 2
    else:
        klist = [1,2,3]
        wcsdim = 3
    
    hdu = fits.PrimaryHDU(output)
    header = hdu.header
    for k in klist:
        header.append(("CDELT"+str(k), 1.))
        header.append(("CD{}_{}".format(k,k), 1.))
        header.append(("LTM{}_{}".format(k,k), 1))
        header.append(("CTYPE"+str(k), "MULTISPE"))
    header.append(("WCSDIM", wcsdim))
    for k, bandid in enumerate(bandids):
        header.append(("BANDID{}".format(k+1), bandid))
    # Do not fill in wavelength/WAT2 yet
    
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(outfname, overwrite=True)

