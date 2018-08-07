import numpy as np
from scipy import special

import time
import pandas as pd
import emcee
#import corner
from astropy import units as u

from pyia import GaiaData
from imp import reload
from . import gaiatools as gtool; reload(gtool)

default_L = 0.5
default_vmax = 1500.
default_alpha = 2.0
default_beta = 8.0

#######################
# Likelihood and prior
def lnlkhd(parallax, pmra, pmdec, cov, d, vtan, phi):
    xmu = np.array([parallax - 1./d,
                    pmra - vtan*np.sin(phi)/(4.74047*d),
                    pmdec- vtan*np.cos(phi)/(4.74047*d)])
    icov = np.linalg.inv(cov)
    arg = -0.5 * xmu.dot(icov.dot(xmu.T))
    return arg - np.log((2*np.pi)**1.5) - 0.5*np.log(np.linalg.det(cov))
def lnprior_d(d,L=default_L):
    """ Expotentially declining prior. d, L in kpc (default L=0.5) """
    if d < 0: return -np.inf
    return -np.log(2) - 3*np.log(L) + 2*np.log(d) - d/L
def lnprior_vtan(vtan,vmax=1500.,alpha=default_alpha,beta=default_beta):
    """ broad velocity prior. Peaks at ~180 km/s with a tail """
    if vtan > vmax or vtan < 0: return -np.inf
    return -special.betaln(alpha,beta) + (alpha-1.)*np.log(vtan/vmax) + (beta-1.)*np.log(1-vtan/vmax)
def lnprior_phi(phi):
    """ flat prior in velocity angle """
    # TODO right now I just let phi float to anywhere
    # This makes chain convergence tests difficult
    return -1.8378770664093464
def lnprior(d,vtan,phi,L=default_L,vmax=default_vmax,alpha=default_alpha,beta=default_beta):
    return lnprior_d(d,L=L) + lnprior_vtan(vtan,vmax=vmax,alpha=alpha,beta=beta) + lnprior_phi(phi)
def lnprob(theta, x, cov, L=default_L, vmax=default_vmax, alpha=default_alpha, beta=default_beta):
    d, vtan, phi = theta
    lp = lnprior(d,vtan,phi,L=L)
    if not np.isfinite(lp): return -np.inf
    parallax, pmra, pmdec = x
    return lp + lnlkhd(parallax, pmra, pmdec, cov, d, vtan, phi)

#######################
# Utility functions
def x_from_theta(theta):
    flip_arr = False
    try:
        d, vtan, phi = theta
    except ValueError:
        assert theta.shape[1] == 3, theta.shape
        theta = theta.T
        d, vtan, phi = theta
        flip_arr = True
    out = np.array([1./d, vtan*np.sin(phi)/(4.74047*d), vtan*np.cos(phi)/(4.74047*d)])
    return out.T if flip_arr else out
def theta_from_x(x):
    parallax, pmra, pmdec = x
    d = 1./parallax
    vtan = 4.74047 * np.sqrt(pmra**2 + pmdec**2) * d
    phi = np.arctan2(pmra,pmdec)
    return np.array([d, vtan, phi])
def _wrap2pi(x):
    if x < 0:
        while x < 0: x += 2*np.pi
        return x
    elif x >= 2*np.pi:
        while x > 2*np.pi: x -= 2*np.pi
        return x
    else:
        return x
wrap2pi = np.vectorize(_wrap2pi)

def generate_parameter_samples(N, theta, cov, apply_filter=True):
    """ Initialize data with a Gaussian"""
    Ndim = len(theta)
    if apply_filter: assert Ndim==3
    if len(cov.shape) == 1: cov = np.diag(cov)
    assert Ndim == cov.shape[1], cov.shape
    samples = np.random.multivariate_normal(theta, cov, size=N)
    
    if apply_filter:
        # require positive d, v, and theta within 2pi
        ii = samples[:,0] < 0
        samples[ii,0] = np.random.uniform(0, theta[0]*2, size=np.sum(ii))
        samples[samples[:,0] < 0,0] = 0.
        
        ii = samples[:,1] < 0
        samples[ii,1] = np.random.uniform(0, theta[1]*2, size=np.sum(ii))
        samples[samples[:,1] < 0,1] = 0.
        
        samples[:,2] = wrap2pi(samples[:,2])
        
    return samples

def guess_dist_vtan(gdat, default_dist=10., default_parallax_snr=1.0,
                    min_derr=0.2):
    """ Guess distance, vtan, and errors with gaussian approximations """
    try:
        dist0 = gdat.distance[0].to('kpc').value
        parallax_over_error = gdat.parallax[0].value/gdat.parallax_error[0].value
    except ValueError: # negative parallax
        dist0 = default_dist
        parallax_over_error = default_parallax_snr
    
    derr0 = max(dist0/parallax_over_error, min_derr)
    vtan0 = dist0 * 4.74047 * np.sqrt(np.sum(gdat.pm[0]**2)).value
    
    eig = np.linalg.eigvals(gdat.get_cov()[0,3:5,3:5])
    verr0a = dist0 * 4.74047 * np.sqrt(np.sum(eig))
    verr0b = derr0 * 4.74047 * np.sqrt(np.sum(gdat.pm[0]**2)).value
    verr0 = np.sqrt(verr0a**2 + verr0b**2)
    
    dv_init = [dist0, vtan0]
    dv_err_init = [derr0, verr0]
    return dv_init, dv_err_init

def sample_distance_vtan(gdat, verbose=True, full_output=False,
                         use_gdat_init=True, default_distance_guess=10.,
                         dv_init=[5., 100.], dv_err_init=[1., 50.], dv_corr_init=0.8,
                         n_walkers=100, n_burn=100, n_chain=1000, n_keep=None, L=default_L,
                         vmax=default_vmax, alpha=default_alpha, beta=default_beta):
    x = np.array([gdat.parallax.value[0], gdat.pmra.value[0], gdat.pmdec.value[0]])
    
    cov = gdat.get_cov()[0,2:5,2:5]
    if use_gdat_init:
        dv_init, dv_err_init = guess_dist_vtan(gdat, default_dist=default_distance_guess)
        if verbose:
            print("Guessing d={:.1f} +/- {:.1f}, vtan={:.1f} +/- {:.1f}".format(
                    dv_init[0], dv_err_init[0], dv_init[1], dv_err_init[1]))
    # If use_gdat_init, already inferred from data
    init_from_data = np.logical_not(use_gdat_init)
    return sample_distance_vtan_mucov(x, cov,
                                      verbose=verbose, full_output=full_output,
                                      init_from_data=init_from_data,
                                      default_distance_guess=default_distance_guess,
                                      dv_init=dv_init, dv_err_init=dv_err_init, 
                                      dv_corr_init=dv_corr_init,
                                      n_walkers=n_walkers, n_burn=n_burn, n_chain=n_chain, 
                                      n_keep=n_keep, L=L, vmax=vmax, alpha=alpha, beta=beta)

def sample_distance_vtan_mucov(x, cov, verbose=True, full_output=False,
                               init_from_data=True, default_distance_guess=10.,
                               dv_init=[5., 100.], dv_err_init=[1., 50.], dv_corr_init=0.8,
                               n_walkers=100, n_burn=100, n_chain=1000, n_keep=None, 
                               L=default_L, vmax=default_vmax, alpha=default_alpha, beta=default_beta):
    """ 
    Run MCMC for a single star. 
    x, cov are mean and covariance matrix in order of [parallax, pmra, pmdec]
    """
    ndim = 3
    x = np.array(x)
    assert len(x) == ndim
    assert cov.shape[0] == ndim
    assert cov.shape[1] == ndim
    
    probfn = lambda *args: lnprob(*args, L=L, vmax=default_vmax, alpha=default_alpha, beta=default_beta)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, probfn, args=(x, cov))
    if verbose: print("{} walkers".format(n_walkers))
    
    ## Initial guess
    if init_from_data:
        default_dist = 10.
        default_parallax_snr = 1.0
        min_derr = 0.2
        if x[0] > 0:
            dist0 = 1./x[0]
            parallax_over_error = x[0]/np.sqrt(cov[0,0])
        else: # default values for negative parallax
            print("Warning: negative parallax, using default values for initialization")
            dist0 = default_dist
            parallax_over_error = default_parallax_snr
        derr0 = max(dist0/parallax_over_error, min_derr)
        vtan0 = dist0 * 4.74047 * np.sqrt(np.sum(x[1:3]**2))
        
        eig = np.linalg.eigvals(cov)
        verr0a = dist0 * 4.74047 * np.sqrt(np.sum(eig))
        verr0b = derr0 * 4.74047 * np.sqrt(np.sum(x[1:3]**2))
        verr0 = np.sqrt(verr0a**2 + verr0b**2)
        
        dv_init = [dist0, vtan0]
        dv_err_init = [derr0, verr0]
        if verbose:
            print("Guessing d={:.1f} +/- {:.1f}, vtan={:.1f} +/- {:.1f}".format(
                    dv_init[0], dv_err_init[0], dv_init[1], dv_err_init[1]))
    
    phi0 = np.arctan2(x[1],x[2])
    print("Initializing with d={:.1f} +/- {:.1f}, vtan={:.1f} +/- {:.1f}, phi={:.0f}deg, dvcorr={:.2f}".format(
           dv_init[0], dv_err_init[0], dv_init[1], dv_err_init[1],phi0*180/np.pi, dv_corr_init))

    theta0 = np.array([dv_init[0], dv_init[1], phi0])
    cov0 = np.array([[dv_err_init[0]**2, dv_err_init[0]*dv_err_init[1]*dv_corr_init, 0],
                     [dv_err_init[0]*dv_err_init[1]*dv_corr_init, dv_err_init[1]**2, 0],
                     [0, 0, (np.pi/2.)**2]])
    pos0 = generate_parameter_samples(n_walkers, theta0, cov0)
    
    ## Burn In
    if verbose: start = time.time()
    pos, prob, stat = sampler.run_mcmc(pos0, n_burn)
    if verbose: print("Burn in {} steps: {:.1f}s".format(n_burn, time.time()-start))
    sampler.reset()
    
    ## Run
    if verbose: start = time.time()
    pos, prob, stat = sampler.run_mcmc(pos, n_chain)
    if verbose:
        print("Chain {} steps: {:.1f}s (accepted {:.3f})".format(n_chain, time.time()-start,
                                                                 np.median(sampler.acceptance_fraction)))
        if np.median(sampler.acceptance_fraction) < 0.3:
            print("WARNING: low acceptance fraction! Chain probably not converged")
    fullchain = sampler.flatchain
    if n_keep is None:
        chain = fullchain
    else:
        if verbose: print("Returning only {} samples".format(n_keep))
        idx_keep = np.random.choice(range(len(fullchain)), size=n_keep, replace=False)
        chain = fullchain[idx_keep, :]

    if verbose:
        print("Results: 90% confidence")
        print("  dist = {:.1f} ({:.1f} - {:.1f})".format(*np.percentile(chain[:,0], [50,5,95])))
        print("  vtan = {:.1f} ({:.1f} - {:.1f})".format(*np.percentile(chain[:,1], [50,5,95])))
    if full_output:
        return chain, sampler, x, cov, theta0, cov0, pos0
    return chain

