# coding: utf-8

""" Misc utility functions """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from six import string_types
import numpy as np
from .robust_polyfit import gaussfit
from scipy import interpolate, signal, stats
import time
from astropy import units
from astropy.stats.biweight import biweight_location, biweight_scale
from astropy import coordinates as coord

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

##def find_distribution_peak(x, x0, s0=None, bins='auto'):
##    """
##    Take a histogram of the data and find the location of the peak closest to x0
##    x: data (no nan's allowed)
##    x0: peak location guess
##    s0: width of peak guess (default std(x))
##    """
##    h, x = np.histogram(x, bins=bins)
##    x = (x[1:]+x[:-1])/2.
##    # find positive peak locations based on derivative
##    dh = np.gradient(h)
##    peaklocs = np.where((dh[:-1] > 0) & (dh[1:] < 0))[0]
##    if len(peaklocs)==0:
##        raise ValueError("No peaks found!")
##    # get the best peak
##    xpeaks = x[peaklocs+1]
##    bestix = np.argmin(np.abs(xpeaks - x0))
##    xbest = x[bestix]
##    ybest = h[bestix]
##    if s0 is None:
##        s_est = np.std(x)
##    else:
##        s_est = s0
##    # fit a Gaussian and get the peak
##    A, xfit, s = gaussfit(x, h, [ybest, xbest, s_est], maxfev=99999)
##    return xfit

def get_cdf_raw(x):
    """
    Take a set of points x and find the CDF.
    Returns xcdf, ycdf, where ycdf = p(x < xcdf)
    Defined such that p(min(x)) = 1/len(x), p(max(x)) = 1
    """
    xcdf = np.sort(x)
    ycdf = np.arange(len(x)).astype(float)/float(len(x))
    return xcdf, ycdf

def find_distribution_mode(x, percentile=5., full_output=False):
    """
    Find the mode of the PDF of a set of elements.
    Algorithm inspired by Shannon Patel
    """
    xcdf, ycdf = get_cdf_raw(x)
    xval = xcdf[1:]
    pdf = np.diff(ycdf)
    # Take the lowest percentile differences, these bunch up near the mode
    pdfcut = np.percentile(pdf, percentile)
    # Take the median of the x's where they bunch up
    return np.median(xval[pdf < pdfcut])

def get_cdf(x, smoothiter=3, window=None, order=1, **kwargs):
    """
    Take a set of points x and find the CDF.
    Use get_cdf_raw, fit and return an interpolating function.
    
    By default we use scipy.interpolate.Akima1DInterpolator
    kwargs are passed to the interpolating function.
    (We do not fit a perfectly interpolating spline right now, because if two points have identical x's,
    you get infinity for the derivative.)
    Note: scipy v1 has a bug in all splines right now that rejects any distribution of points with exactly equal x's.
    
    You can obtain the PDF by taking the first derivative.
    """
    xcdf, ycdf = get_cdf_raw(x)
    # Smooth the CDF
    if window is None:
        window = int(len(xcdf)/100.)
    else:
        window = int(window)
    if window % 2 == 0: window += 1
    F = interpolate.PchipInterpolator(xcdf,ycdf,extrapolate=False)#**kwargs)
    for i in range(smoothiter):
        ycdf = signal.savgol_filter(F(xcdf), window, order)
        F = interpolate.PchipInterpolator(xcdf,ycdf,extrapolate=False)#**kwargs)
    
    #if "ext" not in kwargs:
    #    kwargs["ext"] = 3 #return the boundary value rather than extrapolating
    #F = interpolate.UnivariateSpline(xcdf, ycdf, **kwargs)
    #F = interpolate.Akima1DInterpolator(xcdf,ycdf,**kwargs)
    return F

def find_distribution_peak(x, xtol, full_output=False):
    """
    Finds largest peak from an array of real numbers (x).
    
    Algorithm:
    - Compute the CDF by fitting cubic smoothing spline
    - find PDF with derivative (2nd order piecewise poly)
    - Sample the PDF at xtol/2 points
    - Find peaks in the PDF as the maximum points
    - Return the biggest PDF peak
    """
    Fcdf = get_cdf(x)
    fpdf = Fcdf.derivative()
    
    xsamp = np.arange(np.min(x), np.max(x)+xtol, xtol/2.)
    pdf = fpdf(xsamp)
    ix = np.argmax(pdf)
    xpeak = xsamp[ix]
    
    if full_output:
        return xpeak, xsamp, pdf
    else:
        return xpeak
    
def find_confidence_region(x, p, xtol, full_output=False):
    """
    Finds smallest confidence region from an array of real numbers (x).
    
    Algorithm:
    - use find_distribution_peak to get the peak value and pdf
      The pdf is uniformly sampled at xtol/2 from min(x) to max(x)
    - initialize two tracers x1 and x2 on either side of the peak value
    - step x1 and x2 down (by xtol/2), picking which one locally adds more to the enclosed probability
    
    Note this does not work so well with multimodal things.
    It is also pretty slow, and not so accurate (does not interpolate the PDF or do adaptive stepping)
    """
    assert (p > 0) and (p < 1)
    xpeak, xsamp, pdf = find_distribution_peak(x, xtol, full_output=True)
    
    pdf = pdf / np.sum(pdf) # normalize this to 1
    
    ## Initialize the interval 
    ipeak = np.argmin(np.abs(xsamp-xpeak))
    i1 = ipeak-1; i2 = ipeak+1
    p1 = pdf[i1]; p2 = pdf[i2]
    current_prob = pdf[ipeak]
    def step_left():
        current_prob += p1
        i1 -= 1
        p1 = pdf[i1]
    def step_right():
        current_prob += p2
        i2 += 1
        p2 = pdf[i2]
        
    ## Expand the interval until you get enough probability
    while current_prob < p:
        # If you reached the end, expand the left/right until you're done
        if i1 <= 0:
            while current_prob < p:
                step_right()
            break
        # If you reached the end, expand the left until you're done
        if i2 >= len(pdf):
            while current_prob < p:
                step_left()
            break
        # Step in the direction
        if p1 > p2:
            step_left()
        elif p1 < p2:
            step_right()
        else: # Pick a direction at random if exactly equal
            if np.random.random() > 0.5:
                step_right()
            else:
                step_left()
    if full_output:
        return xsamp[i1], xpeak, xsamp[i2], current_prob, xsamp, pdf
    return xsamp[i1], xpeak, xsamp[i2]

def box_select(x,y,topleft,topright,botleft,botright):
    """
    Select x, y within a box defined by the corner points
    I think this fails if the box is not convex.
    """
    assert len(x) == len(y)
    x = np.ravel(x)
    y = np.ravel(y)
    selection = np.ones_like(x, dtype=bool)
    # Check the directions all make sense
    # I think I assume the box is convex
    assert botleft[1] <= topleft[1], (botleft, topleft)
    assert botright[1] <= topright[1], (botright, topright)
    assert topleft[0] <= topright[0], (topleft, topright)
    assert botleft[0] <= botright[0], (botleft, botright)
    
    # left boundary
    (x1,y1), (x2,y2) = botleft, topleft
    m = (x2-x1)/(y2-y1)
    selection[x < m*(y-y1) + x1] = False
    # right boundary
    (x1,y1), (x2,y2) = botright, topright
    m = (x2-x1)/(y2-y1)
    selection[x > m*(y-y1) + x1] = False

    # top boundary
    (x1,y1), (x2,y2) = topleft, topright
    m = (y2-y1)/(x2-x1)
    selection[y > m*(x-x1) + y1] = False

    # bottom boundary
    (x1,y1), (x2,y2) = botleft, botright
    m = (y2-y1)/(x2-x1)
    selection[y < m*(x-x1) + y1] = False

    return selection

def linefit_2d(x, y, ex, ey, fit_outliers=False, full_output=False,
               nwalkers=20, Nburn=200, Nrun=1000, percentiles=[5,16,50,84,95]):
    """
    Fits a line to a set of data (x, y) with independent gaussian errors (ex, ey) using MCMC.
    Based on Hogg et al. 2010
    
    Returns simple estimate for m, b, and uncertainties.
    
    If fit_outliers=True, fits a background model that is a very flat/wide gaussian.
    Then also returns estimates for 
    
    If full_output=True, return the full MCMC sampler
    
    y = m x + b
    """
    import emcee
    assert len(x)==len(y)==len(ex)==len(ey)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    X = np.vstack([x,y]).T
    ex2 = ex**2
    ey2 = ey**2
    
    # m = tan(theta)
    # bt = b cos(theta)
    if fit_outliers:
        raise NotImplementedError
    else:
        def lnprior(params):
            theta, bt = params
            if theta < -np.pi/2 or theta >= np.pi/2: return -np.inf
            return 0
        def lnlkhd(params):
            theta, bt = params
            v = np.array([-np.sin(theta), np.cos(theta)])
            Delta = X.dot(v) - bt
            Sigma2 = v[0]*v[0]*ex2 + v[1]*v[1]*ey2
            return lnprior(params) - 0.5 * np.sum(Delta**2/Sigma2)
    
    # Initialize walkers
    ymin, ymax = np.min(y), np.max(y)
    bt0 = np.random.uniform(ymin,ymax,nwalkers)
    theta0 = np.random.uniform(-np.pi/2,np.pi/2,nwalkers)
    p0 = np.vstack([theta0,bt0]).T
    
    ndim = 2
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlkhd)
    sampler.run_mcmc(p0,Nburn)
    pos = sampler.chain[:,-1,:]
    sampler.reset()
    sampler.run_mcmc(pos,Nrun)
    
    theta, bt = sampler.flatchain.T
    m = np.tan(theta)
    b = bt/np.cos(theta)
    m_out = np.nanpercentile(m, percentiles)
    b_out = np.nanpercentile(b, percentiles)
    
    if full_output:
        return m_out, b_out, m, b, sampler
    return m_out, b_out

def parse_m2fs_fibermap(fname):
    import pandas as pd
    from astropy.coordinates import SkyCoord
    
    def mungehms(hmsstr,sep=':'):
        h,m,s = hmsstr.split(sep)
        return h+'h'+m+'m'+s+'s'
    def mungedms(dmsstr,sep=':'):
        d,m,s = dmsstr.split(sep)
        return d+'d'+m+'m'+s+'s'
    def parse_assignments(cols, assignments):
        colnames = cols.split()
        data = []
        for line in assignments:
            data.append(line.split())
        return pd.DataFrame(data, columns=colnames)
    
    with open(fname,"r") as fp:
        lines = fp.readlines()
    center_radec = lines[3]
    center_radec = center_radec.split()
    center_ra = mungehms(center_radec[2])
    center_dec= mungedms(center_radec[3])
    center = SkyCoord(ra=center_ra, dec=center_dec)
    for i, line in enumerate(lines):
        if "[assignments]" in line: break
    lines = lines[i+1:]
    line = lines[0]
    cols_assignments = lines[0]
    assignments = []
    for i,line in enumerate(lines[1:]):
        if "]" in line: break
        assignments.append(line)
    lines = lines[i+2:]
    cols_guides = lines[0]
    guides = []
    for i,line in enumerate(lines[1:]):
        if "]" in line: break
        guides.append(line)
    df1 = parse_assignments(cols_assignments, assignments)
    df2 = parse_assignments(cols_guides, guides)
    return df1, df2

def quick_healpix(coo, nside, galactic=False):
    import healpy as hp
    from astropy import units as u
    npix = hp.nside2npix(nside)
    area = hp.nside2pixarea(nside, degrees=True)
    hpmap = np.zeros(npix)
    if galactic:
        theta, phi = np.pi/2 - coo.b.radian, coo.l.wrap_at(180*u.deg).radian
    else:
        theta, phi = np.pi/2 - coo.dec.radian, coo.ra.wrap_at(180*u.deg).radian
    pixels = hp.ang2pix(nside, theta, phi)
    np.add.at(hpmap, pixels, 1)
    hp.mollview(hpmap)
    return hpmap, area

def xbin_yscat(x, y, xbins, q=[2.5,16,50,84,97.5], Nmin=1):
    """
    Take x,y pairs. Bin in x, find percentiles in y.
    
    Input: x and y, xbins
    
    q : default [2.5, 16, 50, 84, 97.5]
        percentiles to compute (passed to np.percentile)
    Nmin : default 1
        minimum number of points per bin to be used (otherwise nan)
    
    Return: xbins centers, ydata percentiles (Nbin x Npercentile)
    """
    assert len(x) == len(y)
    
    xp = (xbins[1:]+xbins[:-1])/2
    Nbins = len(xp)
    ydat = np.zeros((Nbins, len(q))) + np.nan
    
    bin_nums = np.digitize(x, xbins)
    for ibin in range(Nbins):
        bin_num = ibin + 1
        vals = y[bin_nums == bin_num]
        if len(vals) < Nmin: continue
        ydat[ibin, :] = np.percentile(vals, q)
    
    return xp, ydat

def xbin_ybwt(x, y, xbins, Nmin=1):
    """
    Take x,y pairs. Bin in x, find biweight location and scale
    
    Input: x and y, xbins
    
    Nmin : default 1
        minimum number of points per bin to be used (otherwise nan)
    
    Return: xbins centers, yloc, yscale
    """
    assert len(x) == len(y)
    
    xp = (xbins[1:]+xbins[:-1])/2
    Nbins = len(xp)
    yloc = np.zeros(Nbins) + np.nan
    yscale = np.zeros(Nbins) + np.nan
    
    bin_nums = np.digitize(x, xbins)
    for ibin in range(Nbins):
        bin_num = ibin + 1
        vals = y[bin_nums == bin_num]
        if len(vals) < Nmin: continue
        yloc[ibin] = biweight_location(vals)
        yscale[ibin] = biweight_scale(vals)
    
    return xp, yloc, yscale

def xbin_ymean(x, y, xbins, Nmin=1):
    """
    Take x,y pairs. Bin in x, find mean and stdev
    
    Input: x and y, xbins
    
    Nmin : default 1
        minimum number of points per bin to be used (otherwise nan)
    
    Return: xbins centers, ymeans, ystdevs
    """
    assert len(x) == len(y)
    
    xp = (xbins[1:]+xbins[:-1])/2
    Nbins = len(xp)
    ymeans = np.zeros(Nbins) + np.nan
    ystdevs = np.zeros(Nbins) + np.nan
    
    bin_nums = np.digitize(x, xbins)
    for ibin in range(Nbins):
        bin_num = ibin + 1
        vals = y[bin_nums == bin_num]
        if len(vals) < Nmin: continue
        ymeans[ibin] = np.nanmean(vals)
        ystdevs[ibin] = np.nanstd(vals)
    
    return xp, ymeans, ystdevs

def plot_gaussian_distrs(ax, df, xcol, ecol, xplot,
                         plot_individual_stars=True, 
                         color='k', lw=5, ls='-',
                         scale_ind=1.0, ls_ind='-', lw_ind=0.5, label=None):
    """
    Plots the distribution assuming all data is sums of individual little Gaussians
    
    ax: where to plot
    df: data frame
    xcol, ecol: which columns to use for the mean and stdev of each individual datapoint
    xplot: range of x to compute the total 
    plot_individual_stars: if True, plots little gaussians for everything
    
    Other plotting kws:
    label, color, lw, ls
    ls_ind, lw_ind (for individual stars)
    
    """
    x = df[xcol].values
    err = df[ecol].values
    finite = np.isfinite(x) & np.isfinite(err)
    if np.sum(finite) != len(finite):
        print("Dropping {} stars".format(len(finite)-np.sum(finite)))
        print(df.index[~finite])
    xs, errs = x[finite], err[finite]
    N = len(xs)
    all_yplot = np.zeros((len(xplot),N))
    print(len(xplot), all_yplot.shape)
    for i,(x,err) in enumerate(zip(xs,errs)):
        all_yplot[:,i] = stats.norm.pdf(xplot,loc=x,scale=err)
    total_yplot = np.ravel(np.nansum(all_yplot,axis=1))/N
    
    if plot_individual_stars:
        for i in range(N):
            ax.plot(xplot,scale_ind*all_yplot[:,i],'-',ls=ls_ind,lw=lw_ind,color=color,zorder=-9)
    ax.plot(xplot,total_yplot,'-',ls=ls,lw=lw,color=color,zorder=9,label=label)

def get_position_angle(coo1, coo2):
    """
    Based on https://idlastro.gsfc.nasa.gov/ftp/pro/astro/posang.pro
    """
    dRA = coo2.ra - coo1.ra
    numer = np.sin(dRA)
    denom = np.cos(coo1.dec)*np.tan(coo2.dec) - np.sin(coo1.dec)*np.cos(dRA)
    PA = np.arctan2(numer,denom)
    #print(PA)
    if PA < 0: PA += 2*np.pi*units.rad
    return PA

def rv_to_gsr(c, v_sun=None):
    """
    Accessed 2020-12-09
    https://docs.astropy.org/en/stable/generated/examples/coordinates/rv-to-gsr.html
    
    Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

    The input radial velocity must be passed in as a

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()

    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj

def reflex_correct(coords):
    """ https://gala-astro.readthedocs.io/en/latest/_modules/gala/coordinates/reflex.html#reflex_correct """
    galactocentric_frame = coord.Galactocentric()
    c = coord.SkyCoord(coords)

    v_sun = galactocentric_frame.galcen_v_sun

    observed = c.transform_to(galactocentric_frame)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] + v_sun)
    fr = galactocentric_frame.realize_frame(rep).transform_to(c.frame)
    return coord.SkyCoord(fr)
    

def medscat(x):
    med = np.median(x)
    scat = 0.5 * np.sum(np.diff(np.percentile(x, [16, 50, 84])))
    return med, scat


"""
This next part from Sergey Koposov
"""
import astropy.coordinates as acoo
import astropy.units as auni
vlsr0 = 232.8  # from mcmillan 2017

def correct_pm(ra, dec, pmra, pmdec, dist, vlsr=vlsr0, vz=0, split=None):
    if split is None:
        return correct_pm0(ra, dec, pmra, pmdec, dist, vlsr=vlsr0, vz=vz)
    else:
        N = len(ra)
        n1 = N // split

        ra1 = np.array_split(ra, n1)
        dec1 = np.array_split(dec, n1)
        pmra1 = np.array_split(pmra, n1)
        pmdec1 = np.array_split(pmdec, n1)
        if hasattr(dist, '__len__'):
            assert (len(dist) == N)
        else:
            dist = np.zeros(N) + dist
        dist1 = np.array_split(dist, n1)
        ret = []
        for curra, curdec, curpmra, curpmdec, curdist in zip(
                ra1, dec1, pmra1, pmdec1, dist1):
            ret.append(
                correct_pm0(curra,
                            curdec,
                            curpmra,
                            curpmdec,
                            curdist,
                            vlsr=vlsr0))
        retpm1 = np.concatenate([_[0] for _ in ret])
        retpm2 = np.concatenate([_[1] for _ in ret])
        return retpm1, retpm2


def correct_pm0(ra, dec, pmra, pmdec, dist, vlsr=vlsr0, vx=0, vy=0, vz=0):
    """Corrects the proper motion for the speed of the Sun
    Arguments:
        ra - RA in deg
        dec -- Declination in deg
        pmra -- pm in RA in mas/yr
        pmdec -- pm in declination in mas/yr
        dist -- distance in kpc
    Returns:
        (pmra,pmdec) the tuple with the proper motions corrected for the Sun's motion
    """
    C = acoo.ICRS(ra=ra * auni.deg,
                  dec=dec * auni.deg,
                  radial_velocity=0 * auni.km / auni.s,
                  distance=dist * auni.kpc,
                  pm_ra_cosdec=pmra * auni.mas / auni.year,
                  pm_dec=pmdec * auni.mas / auni.year)
    kw = dict(galcen_v_sun=acoo.CartesianDifferential(
        np.array([vx + 11.1, vy + vlsr + 12.24, vz + 7.25]) * auni.km / auni.s))
    frame = acoo.Galactocentric(**kw)
    Cg = C.transform_to(frame)
    Cg1 = acoo.Galactocentric(x=Cg.x,
                              y=Cg.y,
                              z=Cg.z,
                              v_x=Cg.v_x * 0,
                              v_y=Cg.v_y * 0,
                              v_z=Cg.v_z * 0,
                              **kw)
    C1 = Cg1.transform_to(acoo.ICRS())
    return ((C.pm_ra_cosdec - C1.pm_ra_cosdec).to_value(auni.mas / auni.year),
            (C.pm_dec - C1.pm_dec).to_value(auni.mas / auni.year))


def correct_vel(ra, dec, vel, vlsr=vlsr0, vz=0):
    """Corrects the proper motion for the speed of the Sun
    Arguments:
        ra - RA in deg
        dec -- Declination in deg
        pmra -- pm in RA in mas/yr
        pmdec -- pm in declination in mas/yr
        dist -- distance in kpc
    Returns:
        (pmra,pmdec) the tuple with the proper motions corrected for the Sun's motion
    """

    C = acoo.ICRS(ra=ra * auni.deg,
                  dec=dec * auni.deg,
                  radial_velocity=vel * auni.km / auni.s,
                  distance=np.ones_like(vel) * auni.kpc,
                  pm_ra_cosdec=np.zeros_like(vel) * auni.mas / auni.year,
                  pm_dec=np.zeros_like(vel) * auni.mas / auni.year)
    kw = dict(galcen_v_sun=acoo.CartesianDifferential(
        np.array([11.1, vlsr + 12.24, vz + 7.25]) * auni.km / auni.s))
    frame = acoo.Galactocentric(**kw)
    Cg = C.transform_to(frame)
    Cg1 = acoo.Galactocentric(x=Cg.x,
                              y=Cg.y,
                              z=Cg.z,
                              v_x=Cg.v_x * 0,
                              v_y=Cg.v_y * 0,
                              v_z=Cg.v_z * 0,
                              **kw)
    C1 = Cg1.transform_to(acoo.ICRS())
    return np.asarray(((C.radial_velocity - C1.radial_velocity) /
                       (auni.km / auni.s)).decompose())

def get_levels(H, pct = [68,95], zerocut=True):
    """
    Get levels for a contour plot. Default 68 and 95 percent contours.
    """
    HH = np.sort(np.ravel(H))
    if zerocut: HH = HH[HH > 0]
    return(np.percentile(HH, np.array(pct)))
def contour_plot(ax, x, y, xbins, ybins, pct=[68,95], **kwargs):
    """
    Make a contour plot
    """
    H, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins])
    XX, YY = np.meshgrid((xe[1:]+xe[:-1])/2, (ye[1:]+ye[:-1])/2)
    cs = ax.contour(XX, YY, H.T, levels=get_levels(H, pct), **kwargs)
    return cs
