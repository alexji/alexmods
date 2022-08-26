#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Functions related to radial velocity measurement and correction. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["cross_correlate","quick_measure_mike_velocities"]

import logging
import numpy as np
from scipy import interpolate, optimize
from scipy.optimize import leastsq
from astropy.stats.biweight import biweight_scale
import os

from . import spectrum, motions, utils

logger = logging.getLogger(__name__)


def cross_correlate(observed_spectrum, template_spectrum, dispersion_range=None,
    use_weight=False, apodize=0, resample="template", verbose=False, window=None):
    """
    Cross-correlate the observed spectrum against a rest-frame template spectrum
    and measure the radial velocity of the source.

    :param observed_spectrum:
        The observed spectrum.

    :type observed_spectrum:
        :class:`specutils.Spectrum1D`

    :param template_spectrum:
        A rest-frame template spectrum.

    :type template_spectrum:
        :class:`specutils.Spectrum1D`

    :param dispersion_range: [optional]
        A two-length tuple containing the start and end wavelengths to consider.
        If not provided, then the overlap of the `observed_spectrum` and the
        `template_spectrum` will be used.

    :param apodize: [optional]
        The fraction of pixels to apodize on either side of the spectrum.

    :param resample: [optional]
        Whether to resample the 'template' onto the observed spectrum, or
        resample the 'observed' spectrum onto the template.

    :returns:
        The radial velocity, uncertainty in radial velocity, and the CCF.
    """

    if not isinstance(observed_spectrum, spectrum.Spectrum1D):
        raise TypeError(
            "observed_spectrum must be a `specutils.Spectrum1D` object")

    if not isinstance(template_spectrum, spectrum.Spectrum1D):
        raise TypeError(
            "template_spectrum must be a `spectuils.Spectrum1D` object")

    if dispersion_range is None:
        # Use the common ranges.
        dispersion_range = (
            np.max([
                observed_spectrum.dispersion[0],
                template_spectrum.dispersion[0]
            ]),
            np.min([
                observed_spectrum.dispersion[-1],
                template_spectrum.dispersion[-1]
            ])
        )

    if not isinstance(dispersion_range, (tuple, list, np.ndarray)) \
    or len(dispersion_range) != 2:
        raise TypeError("wavelength region must be a two length list-type")

    if apodize != 0:
        raise NotImplementedError("apodization not implemented yet")
        
    resample = resample.lower()
    if resample == "template":
        idx = np.searchsorted(observed_spectrum.dispersion, dispersion_range)
        finite = np.isfinite(observed_spectrum.flux[idx[0]:idx[1]])

        dispersion = observed_spectrum.dispersion[idx[0]:idx[1]][finite]
        observed_flux = observed_spectrum.flux[idx[0]:idx[1]][finite]
        observed_ivar = observed_spectrum.ivar[idx[0]:idx[1]][finite]

        func = interpolate.interp1d(
            template_spectrum.dispersion, template_spectrum.flux,
            bounds_error=False, fill_value=0.0)
        template_flux = func(dispersion)

    elif resample == "observed":
        raise NotImplementedError("why would you do this?")

    else:
        raise ValueError("resample must be 'template' or 'observed'")


    # Perform the cross-correlation
    padding = observed_flux.size + template_flux.size
    # Is this necessary?: # TODO
    x_norm = observed_flux - np.mean(observed_flux[np.isfinite(observed_flux)])
    y_norm = template_flux - np.mean(template_flux[np.isfinite(template_flux)])
    if use_weight:
        x_norm = x_norm * observed_ivar

    Fx = np.fft.fft(x_norm, padding, )
    Fy = np.fft.fft(y_norm, padding, )
    iFxy = np.fft.ifft(Fx.conj() * Fy).real
    varxy = np.sqrt(np.inner(x_norm, x_norm) * np.inner(y_norm, y_norm))

    if use_weight:
        fft_result = iFxy
    else:
        fft_result = iFxy/varxy

    # Put around symmetry axis.
    num = len(fft_result) - 1 if len(fft_result) % 2 else len(fft_result)

    fft_y = np.zeros(num)
    fft_y[:num//2] = fft_result[num//2:num]
    fft_y[num//2:] = fft_result[:num//2]

    fft_x = np.arange(num) - num/2

    # Get initial guess of peak.
    p0 = np.array([fft_x[np.argmax(fft_y)], np.max(fft_y), 10])

    gaussian = lambda p, x: p[1] * np.exp(-(x - p[0])**2 / (2.0 * p[2]**2))
    errfunc = lambda p, x, y: y - gaussian(p, x)

    try:
        p1, ier = leastsq(errfunc, p0.copy(), args=(fft_x, fft_y))

    except:
        logger.exception("Exception in measuring peak of CCF:")
        raise


    # Create functions for interpolating back onto the dispersion map
    fft_points = (0, p1[0], p1[2])
    interp_x = np.arange(num/2) - num/4

    wl_points = []
    for point in fft_points:
        idx = np.searchsorted(interp_x, point)
        try:
            f = interpolate.interp1d(interp_x[idx-3:idx+3], dispersion[idx-3:idx+3],
                bounds_error=True, kind='cubic')
        except ValueError as e:
            print("Interpolation error! Probably bad template? Returning nans with raw CCF")
            print(e)
            print(fft_points, point)
            return np.nan, np.nan, np.array([fft_x, fft_y])
        wl_points.append(f(point))

    # Calculate velocity 
    c = 299792458e-3 # km/s
    f, g, h = wl_points
    rv = c * (1 - g/f)

    # Create a CCF spectrum.
    ccf = np.array([fft_x * (rv/p1[0]), fft_y])

    # Calculate uncertainty
    if use_weight:
        # The ccf should be normalized so that it is close to a chi2 distribution
        # Reducing the ccf by 0.5 from its peak value gives 1-sigma error
        # We approximate this assuming the Gaussian fit is correct
        ymax = p1[1]
        minfunc = lambda x: (ymax - 0.5) - gaussian(p1, x) 
        try:
            xerr, ier = leastsq(minfunc, p1[0] + p1[2])
        except:
            logger.exception("Exception in measuring 1sigma offset for CCF")
            raise
        point = np.abs(xerr - p1[0])
        idx = np.searchsorted(interp_x, point)
        try:
            func = interpolate.interp1d(interp_x[idx-3:idx+3], dispersion[idx-3:idx+3],
                bounds_error=True, kind='cubic')
        except ValueError as e:
            print("Interpolation error in solving for error!")
            print(e, point)
            return np.nan, np.nan, np.array([fft_x, fft_y])
        h = func(point)[0]
        if verbose: print(f,g,h, (h-f)/g, ymax)
        rv_uncertainty = np.abs(c * (h-f)/g)
    else:
        # Approx Uncertainty based on simple gaussian
        # This is not a real uncertainty as it doesn't take into account data errors
        rv_uncertainty = np.abs(c * (h-f)/g)

    return (rv, rv_uncertainty, ccf)



def measure_order_velocities(orders, template, norm_kwargs, **kwargs):
    """
    Run cross correlation against a list of orders
    Return Nx5 array, where columns are order_num, rv, e_rv, wlmin, wlmax
    """
    N = len(orders)
    rv_output = np.zeros((N,5))
    for i, order in enumerate(orders[::-1]):
        if norm_kwargs is None:
            normorder = order
        else:
            normorder = order.fit_continuum(**norm_kwargs)
        try:
            rv, e_rv, ccf = cross_correlate(normorder, template, **kwargs)
        except:
            rv, e_rv = np.nan, np.nan
        try:
            order_num = order.metadata["ECORD{}".format(i)]
        except:
            order_num = i
        rv_output[i,0] = order_num
        rv_output[i,1] = rv
        rv_output[i,2] = e_rv
        rv_output[i,3] = np.min(order.dispersion)
        rv_output[i,4] = np.max(order.dispersion)
    return rv_output

def cross_correlate_2(observed_spectrum, template_spectrum,
                      vmin=-1000., vmax=1000., dv=1,
                      dispersion_range=None,
                      use_weight=False, apodize=0, verbose=False,
                      renormalize_template=False):
    """
    Calculate velocity using naive chi^2.
    Gives a good velocity error estimate.
    
    To mask pixels, put in 0 for ivar in the observed spectrum.

    Returns vfit, err1, err2, voff, chi2arr
    """
    
    wave = observed_spectrum.dispersion
    flux = observed_spectrum.flux
    ivar = observed_spectrum.ivar
    if dispersion_range is not None:
        w1, w2 = dispersion_range
        assert w2 > w1, (w1,w2)
        ii = (wave > w1) & (wave < w2)
        wave = wave[ii]
        flux = flux[ii]
        ivar = ivar[ii]
    else:
        dispersion_range = [wave.min(), wave.max()]
    
    voff = np.arange(vmin,vmax+dv,dv)
    chi2arr = np.zeros_like(voff)
    for i,v in enumerate(voff):
        # shift template by velocity v
        this_template = template_spectrum.copy()
        this_template.redshift(v)
        # interpolate onto new wavelength
        this_template = this_template.linterpolate(wave, fill_value=np.nan)
        if renormalize_template:
            this_template = this_template.cut_wavelength(dispersion_range)
            this_template = this_template.fit_rpoly_continuum()
        Ngood = np.isfinite(this_template.flux).sum()
        # calculate the chi2 of the fit
        if Ngood == 0:
            chi2 = np.inf
        else:
            chi2 = np.nansum(ivar * (flux-this_template.flux)**2.)
        chi2arr[i] = chi2
    vbest = voff[np.argmin(chi2arr)]
    chi2func = interpolate.interp1d(voff, chi2arr, fill_value=np.inf)
    if vbest == voff[0]: vbest = voff[1]
    optres = optimize.minimize_scalar(chi2func, bracket=[voff[0],vbest,voff[-1]])
    if not optres.success:
        print("Warning: optimization did not succeed")
    vfit = optres.x
    chi2min = chi2func(vfit)
    chi2targ = chi2min + 1.0 # single parameter
    
    err1 = optimize.brentq(lambda x: chi2func(x)-chi2targ, voff[0], vfit, xtol=dv/10)
    err2 = optimize.brentq(lambda x: chi2func(x)-chi2targ, vfit, voff[-1], xtol=dv/10)
    
    err1 = err1-vfit
    err2 = err2-vfit
    
    return vfit, err1, err2, voff, chi2arr

def iterative_velocity_measurement(norm, template, wlrange=None,
                                   masks=None,
                                   dvlist=[10,1,.1], vmin=-500., vmax=500., vspanmin=15.):
    """
    Iteratively determine velocity with chi^2 based cross-correlation (cross_correlate_2).
    
    norm = normalized spectrum
    template = normalized template spectrum for RV
    wlrange = used to restrict wavelength range if desired
    dvlist = list of velocity precisions to zoom in by
    vmin, vmax = initial range of velocities to span
    vspanmin = minimum amount of velocity range to calculate error
    """
    chi2min = np.inf
    for dv in dvlist:
        rv, e1, e2, varr, chi2arr = cross_correlate_2(
            norm, template, vmin=vmin, vmax=vmax, dv=dv,
            dispersion_range=wlrange)
        
        vmin = rv + 10*e1
        vmax = rv + 10*e2
        
        if vmax-vmin < vspanmin:
            vmin = rv - vspanmin/2
            vmax = rv + vspanmin/2
        chi2min = min(chi2arr.min(), chi2min)
    return rv, e1, e2, varr, chi2arr, chi2min

def measure_order_velocities_2(orders, template, norm_kwargs,
                               order_min=-np.inf, order_max = np.inf,
                               **kwargs):
    """
    Run cross correlation 2 against a list of orders
    Return Nx5 array, where columns are order_num, rv, e_rv, wlmin, wlmax
    """
    N = len(orders)
    rv_output = np.zeros((N,5))
    ## Orders are opposite ECORD by number so go backwards
    for i, order in enumerate(orders[::-1]):
        if norm_kwargs is None:
            normorder = order
        else:
            normorder = order.fit_continuum(**norm_kwargs)
        
        try:
            order_num = order.metadata["ECORD{}".format(i)]
        except:
            order_num = i
        
        try:
            rv, e_rv1, e_rv2, varr, chi2arr, chi2min = iterative_velocity_measurement(normorder, template, **kwargs)
            e_rv = max(abs(e_rv1), abs(e_rv2))
        except Exception as e:
            print("FAILED at",order_num, e)
            rv, e_rv = np.nan, np.nan
        rv_output[i,0] = order_num
        rv_output[i,1] = rv
        rv_output[i,2] = e_rv
        rv_output[i,3] = np.min(order.dispersion)
        rv_output[i,4] = np.max(order.dispersion)
    return rv_output

def process_rv_output(rv_output):
    """
    Process the output of measure_order_velocities[_2] and find the final velocity + error
    """
    finite = np.isfinite(rv_output[:,1])
    rvdata = rv_output[finite,:]
    raise NotImplementedError

def mask_tellurics(orders, telluric_regions):
    # Mask tellurics
    for order in orders:
        wave = order.dispersion
        ivar = order.ivar
        for wl1,wl2 in telluric_regions:
            ivar[(wl1 < wave) & (wave < wl2)] = 0.
        order._ivar = ivar
    return orders

def quick_measure_mike_velocities(red_fname, template_spectrum=None,
                                  wmin=5150, wmax=5200):
    def _get_overlap_order(input_spectra, wavelength_regions, template_spectrum=None):
        """
        Find the order (and order index) that most overlaps with the template
        spectrum in any of the wavelength_regions provided.

        :param wavelength_regions:
            A list of the wavelength regions to search for overlap between the 
            template spectrum and the input spectra.

        :param template_spectrum: [optional]
            An optional template spectrum that should also overlap with the
            requested ranges. The should be a `specutils.Spectrum1D` object.

        :returns:
            The overlapping order, the overlap index, and the wavelength_region
            that matched.
        """

        # Check to see if wavelength region is a list of entries.
        try:
            int(wavelength_regions[0])
        except (TypeError, ValueError):
            # It is (probably) a list of 2-length tuples.
            None
        else:
            wavelength_regions = [wavelength_regions]

        # Find the order best suitable for the preferred wavelength region.
        for wl_start, wl_end in wavelength_regions:
            # Does the template cover this range?
            if template_spectrum is not None and \
                not (wl_start > template_spectrum.dispersion[0] \
                and  wl_end   < template_spectrum.dispersion[-1]):
                continue

            # Do any observed orders cover any part of this range?
            overlaps, indices = utils.find_overlaps(
                input_spectra, (wl_start, wl_end), return_indices=True)
            if not overlaps:
                continue

            # The first spectral index has the most overlap with the range.
            overlap_index = indices[0]
            overlap_order = overlaps[0]
            break

        else:
            raise ValueError("no wavelength regions are common to the template "
                             "and the observed spectra")

        return (overlap_order, overlap_index, (wl_start, wl_end))
    
    if template_spectrum is None:
        template_spectrum = spectrum.Spectrum1D.read(os.path.join(
            os.path.dirname(__file__), "..", "data/spectra/hd122563.fits"))
    elif isinstance(template_spectrum, spectrum.Spectrum1D):
        pass
    else:
        template_spectrum = spectrum.Spectrum1D.read(template_spectrum)
    
    orders = spectrum.Spectrum1D.read(red_fname)
    overlap_order, overlap_index, wavelength_region = _get_overlap_order(orders, [wmin, wmax], template_spectrum=template_spectrum)
    
    normalization_kwargs = dict(
        #exclude=None,
        #include=None,
        function="spline",
        high_sigma_clip=1.0,
        knot_spacing=20,
        low_sigma_clip=2.0,
        max_iterations=5,
        order=2,
        scale=1.0
    )
    
    observed_spectrum, continuum, _, __ = overlap_order.fit_continuum(
        full_output=True, **normalization_kwargs)

    rv, rv_uncertainty, ccf = cross_correlate(
        observed_spectrum, template_spectrum, wavelength_region, 
        apodize=0, resample="template")
    
    try:
        v_helio, v_bary = motions.corrections_from_headers(overlap_order.metadata)
        v_helio = v_helio.to("km/s").value
        v_bary = v_bary.to("km/s").value
    except Exception as e:
        print(e)
        v_helio = np.nan
    return rv, v_helio

def measure_mike_velocities(template, blue_fname, red_fname,
                            outfname_fig=None, outfname_data=None,
                            vmin=-500, vmax=500, vspanmin=10, dvlist=[10,1,.1],
                            norm_kwargs={"exclude":None, "function":"spline", "high_sigma_clip": 1.0,
                                         "include":None, "knot_spacing": 20, "low_sigma_clip": 2.0,
                                         "max_iterations": 5, "order": 2, "scale": 1.0},
                            telluric_regions=[]):
    
    name = blue_fname.split("/")[-1].split("blue")[0]
    name2 = red_fname.split("/")[-1].split("red")[0]

    if not isinstance(template, spectrum.Spectrum1D):
        # Assume it's a filename to load
        template = spectrum.Spectrum1D.read(template)
    
    ## Blue
    orders = spectrum.Spectrum1D.read(blue_fname)
    orders = mask_tellurics(orders, telluric_regions)
    
    try:
        v_helio, v_bary = motions.corrections_from_headers(orders[0].metadata)
        vhelcorr = v_helio.to("km/s").value
    except Exception as e:
        print("vhel failed:")
        print(e)
        vhelcorr = np.nan

    rv_output1 = measure_order_velocities_2(orders, template, norm_kwargs,
                                            vmin=vmin, vmax=vmax, vspanmin=vspanmin,
                                            dvlist=dvlist)
    
    ## Red
    orders = spectrum.Spectrum1D.read(red_fname)
    orders = mask_tellurics(orders, telluric_regions)
    rv_output2 = measure_order_velocities_2(orders, template, norm_kwargs,
                                            vmin=vmin, vmax=vmax, vspanmin=vspanmin,
                                            dvlist=dvlist)

    
    o_all = np.array(list(rv_output1[:,0]) + list(rv_output2[:,0]))
    v_all = np.array(list(rv_output1[:,1]) + list(rv_output2[:,1]))
    e_all = np.array(list(rv_output1[:,2]) + list(rv_output2[:,2]))
    w_all = np.array(list((rv_output1[:,3]+rv_output1[:,4])/2.) + list((rv_output2[:,3]+rv_output2[:,4])/2.))
    keep = np.isfinite(e_all) & np.isfinite(v_all) & (o_all <= 95) & (o_all >= 50)
    
    for iter_clip in range(5):
        w = e_all[keep]**-2
        v_avg = np.sum(w*v_all[keep])/np.sum(w)
        v_err = (np.sum(w))**-0.5
        v_std = biweight_scale(v_all[keep])
        v_med = np.median(v_all[keep])
        new_keep = keep & (np.abs(v_all-v_med) < 5*v_std)
        print("===============iter_clip={}, {}->{}".format(iter_clip+1,keep.sum(),new_keep.sum()))
        if keep.sum()==new_keep.sum():
            break
        keep = new_keep
    v_blue = np.median(v_all[keep & (o_all > 70)])
    v_red = np.median(v_all[keep & (o_all <= 70)])
    
    if outfname_fig is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,4))
        
        yrange = max(5,5*v_std)
        wave1, wave2 = (rv_output1[:,3]+rv_output1[:,4])/2, (rv_output2[:,3]+rv_output2[:,4])/2
        ax.errorbar(wave1, rv_output1[:,1], yerr=rv_output1[:,2], fmt='o', color='b', ecolor='b')
        ax.errorbar(wave2, rv_output2[:,1], yerr=rv_output2[:,2], fmt='o', color='r', ecolor='r')
        ax.plot(w_all[keep], v_all[keep], 'ko', mfc='none', mec='k', mew=2, ms=10)
        
        ax.set_ylim(v_avg - yrange, v_avg + yrange)         
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        ax.axhline(v_avg, color='k', zorder=-9)
        
        fig.tight_layout()
        fig.savefig(outfname_fig)
        plt.close(fig)

    if outfname_data is not None:
        np.save(outfname_data, [(v_avg, v_med, v_err, v_std, v_blue, v_red, vhelcorr), (o_all, v_all, e_all, w_all, keep), rv_output1, rv_output2])
    
    return v_avg, v_med, v_err, v_std, v_blue, v_red, vhelcorr
