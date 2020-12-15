import numpy as np
import pandas as pd
from scipy import interpolate
import warnings

import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import table

import gaia_tools.load as gload
from pyia import GaiaData

def load_tgas():
    """
    Creates pyia.GaiaData object from TGAS (a subclass of pandas DataFrame)
    """
    tgas = GaiaData(gload.tgas())
    return tgas


##############################################
## Columns from Gaia DR2 data model 
## https://www.cosmos.esa.int/documents/29201/1645651/GDR2_DataModel_draft.pdf/938f48a2-a08d-b63c-67e7-eae778c9a657
##############################################
cols_astrometry = "ra,dec,parallax,pmra,pmdec"
ecol_astrometry = "ra_error,dec_error,parallax_error,parallax_over_error,"+\
                  "pmra_error,pmdec_error,ra_dec_corr,ra_parallax_corr,ra_pmra_corr,"+\
                  "ra_pmdec_corr,dec_parallax_corr,dec_pmra_corr,dec_pmdec_corr,parallax_pmra_corr,"+\
                  "parallax_pmdec_corr,pmra_pmdec_corr,duplicated_source"
qual_astrometry = "astrometric_n_obs_al,astrometric_n_obs_ac,astrometric_n_good_obs_al,astrometric_n_bad_obs_al,"+\
                  "astrometric_gof_al,astrometric_chi2_al,astrometric_excess_noise,astrometric_excess_noise_sig,"+\
                  "astrometric_params_solved,astrometric_primary_flag,astrometric_weight_al,"+\
                  "astrometric_pseudo_colour,astrometric_pseudo_colour_error,"+\
                  "mean_varpi_factor_al,astrometric_matched_observations,visibility_periods_used,"+\
                  "astrometric_sigma5d_max,frame_rotator_object_type,matched_observations"
cols_phot = "phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,phot_variable_flag"
ecol_phot = "phot_g_mean_flux,phot_bp_mean_flux,phot_rp_mean_flux,"+\
            "phot_g_mean_flux_error,phot_g_mean_flux_over_error,"+\
            "phot_bp_mean_flux_error,phot_bp_mean_flux_over_error,"+\
            "phot_rp_mean_flux_error,phot_rp_mean_flux_over_error"
qual_phot = "phot_g_n_obs,phot_bp_n_obs,phot_rp_n_obs,phot_bp_rp_excess_factor,phot_proc_mode"
cols_redd = "bp_rp,bp_g,g_rp,a_g_val,e_bp_min_rp_val,"+\
             "a_g_percentile_lower,a_g_percentile_upper,"+\
             "e_bp_min_rp_percentile_lower,e_bp_min_rp_percentile_upper"
cols_spec = "radial_velocity,radial_velocity_error"
qual_spec = "rv_template_teff,rv_template_logg,rv_template_fe_h,rv_nb_transits"
cols_star = "teff_val,radius_val,lum_val"
ecol_star = "teff_percentile_lower,teff_percentile_upper,"+\
            "radius_percentile_lower,radius_percentile_upper,"+\
            "lum_percentile_lower,lum_percentile_upper"
cols_rave = ""
ecol_rave = ""

all_columns = ",".join(["source_id", cols_astrometry, ecol_astrometry, qual_astrometry,
                         cols_phot, ecol_phot, qual_phot, cols_redd, cols_spec, qual_spec, cols_star, ecol_star])
## This is a full set of things that I think will be useful
full_columns = ",".join(["source_id", cols_astrometry, ecol_astrometry,
                         cols_phot, ecol_phot, cols_redd, cols_spec, qual_spec, cols_star, ecol_star])
## This is a minimal set of things that I think will be useful
default_columns = ",".join(["source_id",cols_astrometry,ecol_astrometry,
                            cols_phot, cols_spec, cols_star])
                         

def create_source_query_from_ids(ids, columns=default_columns,
                                 source="gaiaedr3.gaia_source"):
    out = "SELECT {} FROM {} WHERE ".format(
        columns, source)
    idstrs = " or ".join(["source_id = {}".format(x) for x in ids])
    out += idstrs
    return out

def create_source_query_from(coords, radius=1*u.arcsec,
                             columns=default_columns,
                             source="gaiaedr3.gaia_source",
                             Nmax=None):
    """
    Generate a string selecting specific list of coordinates.
    Built from https://gist.github.com/mfouesneau/b6b25ed645eab9da4710153fcf9a4cb8
    """
    N = len(coords)
    if Nmax is None: Nmax = 2*N
    out = "SELECT TOP {} {} FROM {} WHERE ".format(
        Nmax, columns, source)
    def _make_contains_str(c):
        cstr = "CONTAINS(POINT('ICRS',{0:}.ra,{0:}.dec),CIRCLE('ICRS',{1:},{2:},{3:}))=1".format(
            source, c.ra.deg, c.dec.deg, radius.to("deg").value)
        return cstr
    cstrs = map(_make_contains_str, coords)
    out += " or ".join(cstrs)
    return out

def create_samples(Nsamp,mu,cov):
    Nstars,Nparams = mu.shape
    assert Nstars == len(cov)
    assert Nparams == cov.shape[1]
    output = np.zeros((Nsamp*Nstars, Nparams))
    for i in range(Nstars):
        i1 = Nsamp*i
        i2 = Nsamp*(i+1)
        output[i1:i2,:] = np.random.multivariate_normal(mu[i,:],cov[i,:,:],Nsamp)
    output = output.reshape(Nstars, Nsamp, Nparams)
    return output

def get_gc_frame():
    v_sun = coord.CartesianDifferential([11.1, 250, 7.25]*u.km/u.s)
    #gc_frame = coord.Galactocentric(galcen_distance=8.3*u.kpc,
    #                                z_sun=0*u.pc,
    #                                galcen_v_sun=v_sun)
    gc_frame = coord.Galactocentric()
    return gc_frame
def get_gccoo_w0(coo):
    gc_frame = get_gc_frame()
    gccoo = coo.transform_to(gc_frame)
    w0 = gd.PhaseSpacePosition(gccoo.data)
    return gccoo, w0

def get_orbit_params(orbits):
    N = orbits.shape[1]
    pers = []
    apos = []
    eccs = []
    for i in range(N):
        orbit = orbits[:,i]
        rp, ra = orbit.pericenter(), orbit.apocenter()
        pers.append(rp)
        apos.append(ra)
        eccs.append((ra - rp) / (ra + rp))
    return u.Quantity(pers), u.Quantity(apos), u.Quantity(eccs)

def get_orbit_params_fast(orbits):
    try:
        N = orbits.shape[1]
    except IndexError:
        orbit = orbits
        r = np.sqrt(np.sum(orbits.xyz**2,axis=0))
        rp, ra = np.min(r), np.max(r)
        return u.Quantity(rp), u.Quantity(ra), u.Quantity((ra-rp)/(ra+rp))
    pers = []
    apos = []
    eccs = []
    for i in range(N):
        orbit = orbits[:,i]
        r = np.sqrt(np.sum(orbit.xyz**2,axis=0))
        rp, ra = np.min(r), np.max(r)
        pers.append(rp)
        apos.append(ra)
        eccs.append((ra - rp) / (ra + rp))
    return u.Quantity(pers), u.Quantity(apos), u.Quantity(eccs)

def calc_vtan_error(pmra, pmdec, parallax):
    d = u.kpc / parallax.value
    pmra = pmra.to(u.rad/u.yr, u.dimensionless_angles())
    pmdec= pmdec.to(u.rad/u.yr, u.dimensionless_angles())
    vtan = d * np.sqrt(pmra**2 + pmdec**2)
    vtan = vtan.to(u.km/u.s, u.dimensionless_angles())
    return vtan
    
def avgstd(x,ignore_nan=False, axis=None):
    mean = np.nanmean if ignore_nan else np.mean
    stdev = np.nanstd if ignore_nan else np.std
    kws = {}
    if axis is not None: kws['axis'] = axis
    mu = mean(x,**kws)
    sig = stdev(x,**kws)
    return np.vstack([mu,sig]).T
    
def medscat(x,sigma=2,ignore_nan=False, axis=None, for_errorbar_plot=False):
    percentile = np.nanpercentile if ignore_nan else np.percentile
    pdict = {1:[16,50,84],2:[5,50,95],3:[.1,50,99.9]}
    assert sigma in pdict
    
    kws = {}
    if axis is not None: kws['axis'] = axis
    p1,p2,p3 = percentile(x, pdict[sigma], **kws)
    e1 = p1-p2
    e2 = p3-p2
    if for_errorbar_plot:
        e1 = -e1
        return p2, np.stack([e1,e2])
    return np.stack([e1,p2,e2])
    
def modefinder(x, bins="auto", dropna=True):
    """
    Estimates the mode of a sample of points.
    Assumes a unimodal system.
    
    Take a histogram of the data and return the bin with the largest value.
    TODO If an initial value is specified, find the local maximum closest to that value.
    """
    if dropna: x = x[np.isfinite(x)]
    
    h,x = np.histogram(x, bins=bins)
    xm = (x[1:]+x[:-1])/2.
    ix = np.argmax(h)
    return xm[ix]

def get_finite(x,y):
    """ Get x and y that are both finite """
    finite = np.logical_and(np.isfinite(x), np.isfinite(y))
    xf = x[finite]; yf = y[finite]
    return xf, yf

def fit_spline(x, y, **kwargs):
    """ A simple wrapper to scipy.interpolate.UnivariateSpline (remove nan, sort x) """
    xf, yf = get_finite(x,y)
    iisort = np.argsort(xf)
    return interpolate.UnivariateSpline(xf[iisort],yf[iisort], **kwargs)

def bin_medscat(x, y, percentiles=[5,50,95], for_errorbar_plot=False, dropna=True, bins="auto", **kwargs):
    """
    Histogram x into bins.
    Then in those bins, take percentiles of y.
    """
    if dropna: x, y = get_finite(x, y)
    h, xe = np.histogram(x, bins=bins, **kwargs)
    xout = (xe[1:]+xe[:-1])/2.
    indices = np.digitize(x, xe)
    yout = np.zeros((len(xe)-1,len(percentiles)))+np.nan
    for ix in np.unique(indices):
        # Skip things outside the bin range
        if ix >= len(yout): continue
        # Percentile in this bin
        ii = ix==indices
        yout[ix,:] = np.percentile(y[ii], percentiles)
    if for_errorbar_plot:
        e1 = yout[:,1] - yout[:,0]
        e2 = yout[:,2] - yout[:,1]
        return xout, yout[:,1], [e1,e2]
    return xout, yout

def calculate_actions(w0,pot=gp.MilkyWayPotential(), dt=0.5, n_steps=10000, full_output=False):
    """ Approximate actions following https://github.com/adrn/gala/blob/master/docs/dynamics/actionangle.rst """
    assert len(w0.shape)==0
    w = gp.Hamiltonian(pot).integrate_orbit(w0, dt=dt, n_steps=n_steps)
    toy_potential = gd.fit_isochrone(w)
    toy_actions, toy_angles, toy_freqs = toy_potential.action_angle(w)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        result = gd.find_actions(w, N_max=8, toy_potential=toy_potential)
    if full_output: return result, w
    return result["actions"]

def query_and_match(coo, match_radius=1, columns=full_columns):
    """
    Query gaia given coordinates
    Return a table that is sorted, and an array saying which rows actually matched an object in gaia
    """
    from pyia import GaiaDataNew
    query = create_source_query_from(coo, columns=columns)
    gaia = GaiaDataNew.from_query(query)
    gcoo = SkyCoord(gaia.ra, gaia.dec)
    idx, d2d, _ = coo.match_to_catalog_sky(gcoo)
    iimatch = d2d.arcsec < match_radius
    gtab = gaia.data[idx]
    if iimatch.sum() != len(gtab):
        print("Warning: only matched {}/{} stars".format(iimatch.sum(),len(gtab)))
    return gtab, iimatch

def query_and_match_sourceid(source_ids, match_radius=1, columns=full_columns):
    """
    Query gaia given source_ids
    Return a table in the order of the source_ids
    """
    from pyia import GaiaDataNew
    unique_arr, indexes = np.unique(source_ids, return_inverse=True)
    assert len(unique_arr) == len(source_ids), "Not all IDs are unique"
    query = create_source_query_from_ids(source_ids, columns=columns)
    gaia = GaiaDataNew.from_query(query)
    # Sort by source id, find indices, then resort
    gdat = gaia.data
    gdat.sort("source_id")
    assert np.all(unique_arr == gdat["source_id"])
    gdat = gdat[indexes]
    assert np.all(gdat["source_id"]==source_ids)
    return gdat
