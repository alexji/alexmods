# coding: utf-8
""" Some photometry tools for stellar spectroscopists """
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from scipy import interpolate
from astropy.io import ascii
from .robust_polyfit import polyfit

import logging
import os, sys, time
logger = logging.getLogger(__name__)

__all__ = []

from .read_data import datapath
from .read_data import load_parsec_isochrones, load_dartmouth_isochrones


def eval_BC(Teff,logg,FeH,filt="g",allBCs=None):
    """
    Default is alpha/Fe = +0.4
    """
    if allBCs is None: allBCs = read_bc_table()
        
    BCs = allBCs[filt]
    
    points = np.atleast_2d([np.ravel(Teff),np.ravel(logg),np.ravel(FeH)]).T
    points[points[:,2] < -2.5,2] = -2.5
    out = interpolate.griddata(BCs[:,0:3], BCs[:,3], points, method='linear')
    return out
def read_bc_table(fname=datapath+"/bolometric_corrections/bc_p04_ugriz.data"):
    """
    Load a Casagrande+Vandenberg 2014 BC table
    """
    with open(fname,'r') as fp:
        lines = fp.readlines()
    s = lines[1].split()
    NTeff, Nlogg, NMH, Nfilt = int(s[0]), int(s[2]), int(s[5]), int(s[7])
    allBCs = {}

    Teffs = list(map(float, "".join(lines[2:5]).replace("\n"," ").split()))
    loggs = list(map(float, lines[5].split()))
    Nlist = list(map(int, lines[6].split()))
    iline = 7
    allBCs = {}
    for ifilt in range(Nfilt):
        BCtable = np.zeros((np.sum(Nlist)*NMH,4))
        itable = 0
        for iMH in range(NMH):
            s = lines[iline].split()
            FeH = float(s[2]); aFe = float(s[5]); filter = s[9]
            iline += 1
            for ilogg,logg in enumerate(loggs):
                BCrow = []
                while len(BCrow) < Nlist[ilogg]:
                    line = lines[iline]
                    iline += 1
                    BCrow += list(map(float, line.split()))
                for iTeff,Teff in enumerate(Teffs[0:Nlist[ilogg]]):
                    BCtable[itable,0] = Teff
                    BCtable[itable,1] = logg
                    BCtable[itable,2] = FeH
                    BCtable[itable,3] = BCrow[iTeff]
                    itable += 1
        allBCs[filter] = BCtable
    return allBCs

##################################################################
# From Drlica-Wagner et al. 2018 (https://arxiv.org/abs/1708.01531)
# g_{des} = g_{sdss} - 0.104 \times (g-r)_{sdss} + 0.01
# r_{des} = r_{sdss} - 0.102 \times (g-r)_{sdss} + 0.02
# i_{des} = i_{sdss} - 0.256 \times (i-z)_{sdss} + 0.02
# z_{des} = z_{sdss} - 0.086 \times (i-z)_{sdss} + 0.01
##################################################################
def gr_sdss2des(gsdss,rsdss):
    gmrsdss = gsdss - rsdss
    gdes = gsdss - 0.104 * gmrsdss + 0.01
    rdes = rsdss - 0.102 * gmrsdss + 0.02
    return gdes, rdes
def iz_sdss2des(isdss,zsdss):
    imzsdss = isdss - zsdss
    ides = isdss - 0.256 * imzsdss + 0.02
    zdes = zsdss - 0.086 * imzsdss + 0.01
    return ides, zdes
def gr_des2sdss(gdes,rdes):
    gmrdes = gdes-rdes
    gmrsdss = (gmrdes + 0.01)/0.998
    gsdss = gdes + 0.104 * gmrsdss - 0.01
    rsdss = rdes + 0.102 * gmrsdss - 0.02
    return gsdss, rsdss
def iz_des2sdss(ides,zdes):
    imzdes = ides-zdes
    imzsdss = (imzdes - 0.01)/0.830
    isdss = ides + 0.256 * imzsdss - 0.02
    zsdss = zdes + 0.086 * imzsdss - 0.01
    return isdss, zsdss
def griz_des2sdss(gdes,rdes,ides,zdes):
    gsdss, rsdss = gr_des2sdss(gdes,rdes)
    isdss, zsdss = iz_des2sdss(ides,zdes)
    return gsdss, rsdss, isdss, zsdss

### Setup Jordi06
# http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Jordi2006
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

def jordi06_gmi_to_VmI(gmi,geterr=True):
    assert np.all(np.logical_or(np.ravel(gmi) < 2.1, np.isnan(np.ravel(gmi))))
    VmI = 0.674 * gmi + 0.406
    if geterr:
        VmImin = (0.674-0.005)*gmi + (0.406 - 0.004)
        VmImax = (0.674+0.005)*gmi + (0.406 + 0.004)
        return VmImin, VmI, VmImax
    return VmI
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

###################################################################
# From Casagrande et al. 2010, applicable to dwarfs and subgiants #
###################################################################
def C10_Teff_BmV(BmV, FeH):
    """ 73K scatter """
    a0, a1, a2, a3, a4, a5 = .5665, .4809, -.0060, -.0613, -.0042, -.0055
    theta = a0 + a1*BmV + a2*BmV*BmV + a3*BmV*FeH + a4*FeH + a5*FeH*FeH
    Teff = 5040./theta
    return Teff
def C10_Teff_VmI(VmI, FeH):
    """ 59K scatter """
    a0, a1, a2, a3, a4, a5 = .4033, .8171, -.1987, -.0409, .0319, .0012
    theta = a0 + a1*VmI + a2*VmI*VmI + a3*VmI*FeH + a4*FeH + a5*FeH*FeH
    Teff = 5040./theta
    return Teff
##################################
# From Alonso et al. 1999: F0-K5 #
##################################
def A99_BC_V(Teff, FeH):
    """
    Typical scatter is 0.025 for cool stars, 0.009 for warm stars (dividing at T=4500K)
    Limits of applicability are 3.5 < logT < 3.96, though different for different [Fe/H] ranges
    """
    X = np.ravel(np.log10(Teff) - 3.52); FeH = np.ravel(FeH)
    # Equations 17 and 18
    BC17 = -5.531e-2/X - 0.6177 + 4.420*X - 2.669*X**2. + 0.6943*X*FeH - 0.1071*FeH - 8.612e-3*FeH**2.
    BC18 = -9.930e-2/X + 2.887e-2 + 2.275*X - 4.425*X**2. + 0.3505*X*FeH - 5.558e-2*FeH - 5.375e-3*FeH**2
    BC = BC17.copy()
    ii = np.log10(Teff) >= 3.65
    BC[ii] = BC18[ii]
    return BC
    
def B79_VmI_C2J(VmI):
    """ Convert V-I in Cousins' mags to V-I in Johnson's mags from Bessell 1979 """
    VmI = np.ravel(VmI)
    out = VmI.copy()/0.778
    out[VmI < 0] = VmI[VmI < 0]/0.713
    ii = out > 2.0
    out[ii] = (VmI[ii]+0.13)/0.835
    return out
def A99_Teff_VmI(VmI):
    """
    Johnson's V, Johnson's (NOT Cousins') I
    125K scatter, no dependence on Fe/H.
    I have assumed that VmI is given in Johnson-Cousins, and 
    """
    VmI = B79_VmI_C2J(VmI)
    theta = 0.5379 + 0.3981 * VmI + 4.432e-2 * VmI**2 - 2.693e-2 * VmI**3
    Teff = 5040./theta
    return Teff

def _A99_function(X, FeH, a0, a1, a2, a3, a4, a5):
    return a0 + a1*X + a2*X**2. + a3*X*FeH + a4*FeH + a5*FeH**2.
def _A99_Teff_BmV_3(BmV, FeH):
    """ 167K scatter, B-V < 0.7 """
    a0, a1, a2, a3, a4, a5 = 0.5716, 0.5404, -6.126e-2, -4.862e-2, -1.777e-2, -7.969e-3
    return _A99_function(BmV, FeH, a0, a1, a2, a3, a4, a5)
def _A99_Teff_BmV_4(BmV, FeH):
    """ 96K scatter, B-V > 0.8 """
    a0, a1, a2, a3, a4, a5 = 0.6177, 0.4354, -4.025e-3, 5.204e-2, -0.1127, -1.385e-2
    return _A99_function(BmV, FeH, a0, a1, a2, a3, a4, a5)
def A99_Teff_BmV(BmV, FeH):
    """
    Johnson's B and V
    Using equations 3 and 4 of A99, scatter is 167K
    Linearly interpolating in theta = 5040/Teff for 0.7 < B-V < 0.8
    """
    BmV = np.ravel(BmV); FeH = np.ravel(FeH)
    t3 = _A99_Teff_BmV_3(BmV, FeH)
    t4 = _A99_Teff_BmV_4(BmV, FeH)
    # Bluest stars, Eq 3
    t = t3.copy()
    # Reddest stars, Eq 4
    t[BmV > 0.8] = t4[BmV > 0.8]
    # In between: 0.7 < B-V < 0.8, linear interpolate
    ii = np.logical_and(BmV > 0.7, BmV <= 0.8)
    x1, x2 = 0.7, 0.8
    y1 = _A99_Teff_BmV_3(x1, FeH)
    y2 = _A99_Teff_BmV_4(x2, FeH)
    m = (y2 - y1)/(x2 - x1)
    y = m * (BmV - x1) + y1
    t[ii] = y[ii]
    return 5040./t

def phot_logg(Teff,mag0,BCmag,distmod,Mstar=0.75):
    """
    Using solar values from Venn et al. 2017
    """
    return 4.44 + np.log10(Mstar) + 4*np.log10(Teff/5780) + 0.4 * (mag0 - distmod + BCmag - 4.75)
def iterate_find_logg(Teff,mag0,FeH,dmod,filt,maxiter=10,tol=.005):
    """ Assumes [alpha/Fe] = +0.4, sdss mags for filt """
    # Initialize BC and logg
    BC = 0.0
    logg = phot_logg(Teff,mag0,BC,dmod)
    for iter in range(maxiter):
        BC = eval_BC(Teff, logg, FeH, filt=filt)
        new_logg = phot_logg(Teff,mag0,BC,dmod)
        if np.all(np.abs(new_logg - logg) < tol):
            break
        logg = new_logg
    else:
        print("WARNING: Reached max iters")
    return logg
def phot_logg_error(Tfracerr, dmoderr, masserr=0.05, magerr=0.0, BCerr=0.03):
    """
    Estimate 1 sigma error in logg 
    Tfracerr: temperature error divided by temperature
    dmoderr: distance modulus error in mag
    masserr (0.05 mag): from assuming a mass, 0.05 is 0.7-0.8 Msun
    magerr: assume this is negligible by default
    BCerr: estimated about 0.03 mag from running CV14 several times
    """
    Terr_mag = 4*Tfracerr # from a taylor expansion
    magerr = 0.4*magerr
    BCerr = 0.4*BCerr
    dmoderr = 0.4*dmoderr
    return np.sqrt(masserr**2 + Terr_mag**2 + magerr**2 + dmoderr**2 + BCerr**2)

###################
## Y2 isochrones ##
###################
def get_logT_to_logg(FeH=-3.0):
    assert FeH in [-2.0, -2.5, -3.0]
    if FeH == -2.0:
        iso = ascii.read(datapath+'/stellar_param_data/afe040feh200set1_12gyr.txt')
    elif FeH == -2.5:
        iso = ascii.read(datapath+'/stellar_param_data/afe040feh250set1_12gyr.txt')
    elif FeH == -3.0:
        iso = ascii.read(datapath+'/stellar_param_data/afe040feh300set1_12gyr.txt')
    ii_max_logT = np.argmax(iso['logT'])
    max_logT = iso[ii_max_logT]['logT']
    max_logg = iso[ii_max_logT]['logg']
    #print max_logT, max_logg
    ii = iso['logg'] < max_logg
    logT = iso[ii]['logT']
    logg = iso[ii]['logg']
    logT_to_logg = interpolate.interp1d(logT,logg)
    return logT_to_logg
_my_interps = [get_logT_to_logg(FeH) for FeH in [-2.0,-2.5,-3.0]]
def _logTFeH_to_logg(logT,FeH):
    if FeH > -2.0: return _my_interps[0](logT)
    elif FeH <= -3.0: return _my_interps[2](logT)
    elif FeH <= -2.0 and FeH > -2.5:
        x = (FeH+2.5)*2.0
        assert x <= 1 and x >= 0
        logg1 = _my_interps[0](logT)
        logg2 = _my_interps[1](logT)
        return logg1 * x + logg2 * (1-x)
    elif FeH <= -2.5 and FeH > -3.5:
        x = (FeH+3.0)*2.0
        assert x <= 1 and x >= 0
        logg1 = _my_interps[1](logT)
        logg2 = _my_interps[2](logT)
        return logg1 * x + logg2 * (1-x)
    else:
        raise ValueError("FeH = {}".format(FeH))
logTFeH_to_logg = np.vectorize(_logTFeH_to_logg)
###############################
## Microturbulence Relations ##
###############################
def get_logg_to_vt_B05():
    b = ascii.read(datapath+'/stellar_param_data/barklem.txt')
    ## This fails in the newer version of scipy
    #iisort = np.argsort(b['logg'])
    #fit = interpolate.UnivariateSpline(b['logg'][iisort],b['Vt'][iisort],k=2)
    coeff, sigma = polyfit(b['logg'],b['Vt'],2)
    fit = lambda x: np.polyval(coeff, x)
    return fit
def logg_to_vt_B05(logg):
    fit = get_logg_to_vt_B05()
    return fit(logg)
def logg_to_vt_K09(logg):
    """ Kirby et al. 2009 ApJ 705, 328 (uncertainty is ~ 0.05 + 0.03*logg) """
    return 2.13 - 0.23 * logg
def logg_to_vt_M08(logg):
    """ Marino et al. 2008 A&A 490, 625 (from Gratton et al. 1996) """
    return 2.22 - 0.322 * logg


#################
## Dereddening ##
#################
def deredden(EBV,filt):
    """ Subtract this value from the observed magnitude to get the dereddened mags """
    conversion_data = ascii.read(datapath+"/stellar_param_data/sf11.txt")
    assert filt in conversion_data["filter"], (filt, conversion_data["filter"])
    return EBV * float(conversion_data["AB_EBV"][np.where(conversion_data["filter"]==filt)[0]])

"""
Notes about filter conversions and definitions.
Johnson-Cousins system: UBV in Johnson, RI in Cousins. I think this is the same as the Landolt system.

Jordi+2006: converts from SDSS (as observed at APO, with primes???) to UBV(RI)c.
Alonso+1999: converts JOHNSON'S ONLY colors to Teff. So RI need to go to (RI)c if you use V-I.
Casagrande+2010: converts Johnson-Cousins to Teff

So the most consistent thing for DES mags is to go from griz_DES -> griz_SDSS -> UBV(RI)c -> Casagrande+2010
Note Casagrande+2010 is not calibrated to very red giants (<4500K).

For E(B-V)=0.02, I found the order in which you deredden makes <1 mmag difference in the final color.
"""


def determine_stellar_params(gmag,rmag,imag,zmag,
                             MH,dmod,
                             EBV=0,EBVerr=0.0,dmoderr=0.1,
                             gerr=0.02,rerr=0.02,ierr=0.02,zerr=0.02,
                             verbose=True,fp=sys.stdout,
                             Teff_color="VmI", Teff_calib="C10",
                             logg_mag="r", full_output=False):
    """
    [g,r,i,z]mag: DES magnitudes
    MH: input metallicity
    dmod: distance modulus
    [g,r,i,z]err: magnitude error 
        default 0.02 mag in each band, the absolute calibration uncertainty (ADW+2017 arxiv:1708.01531)
        (The internal calibration uncertainty is <4mmag)
    
    Effective temperature error includes:
      [g/r/i]err, EBVerr, Jordi06 err
    """
    
    assert Teff_color in ["BmV","VmI"], Teff_color
    assert Teff_calib in ["C10","A99"], Teff_calib
    assert logg_mag in ["g","r","i"], logg_mag
    
    out = griz_des2sdss(gmag,rmag,imag,zmag)
    g,r,i,z = out
    if verbose:
        fp.write("g-r={:.2f}->{:.2f}\n".format(gmag-rmag,g-r))
        fp.write("g-i={:.2f}->{:.2f}\n".format(gmag-imag,g-i))
    
    logg_mag_dict = {"g":g,"r":r,"i":i}
    logg_magerr_dict = {"g":gerr,"r":rerr,"i":ierr}
    
    ## Determine Effective Temperature and Error
    ## Output: Teff, Teff_err, color, color_err
    if Teff_color=="BmV":
        BmV1, BmV, BmV2 = jordi06_gmr_to_BmV(g-r, geterr=True) 
        BmVerr = max(abs(BmV2-BmV), abs(BmV-BmV1))
        BmVerr = np.sqrt(BmVerr**2. + gerr**2 + rerr**2 + EBVerr**2)
        BmV = BmV + EBV
        if Teff_calib=="C10":
            Teff = C10_Teff_BmV(BmV, MH)
            Teff1 =  C10_Teff_BmV(BmV-BmVerr, MH)
            Teff2 =  C10_Teff_BmV(BmV+BmVerr, MH)
            Teff_syserr = 73.
        elif Teff_calib=="A99":
            Teff = A99_Teff_BmV(BmV, MH)
            Teff1 = A99_Teff_BmV(BmV-BmVerr, MH)
            Teff2 = A99_Teff_BmV(BmV+BmVerr, MH)
            Teff_syserr = 167.
        color_err = BmVerr
        color = BmV
    elif Teff_color=="VmI":
        EVI = deredden(EBV, "LandoltV") - deredden(EBV, "LandoltI")
        VmI1, VmI, VmI2 = jordi06_gmi_to_VmI(g-i, geterr=True)
        VmIerr = max(abs(VmI2 - VmI), abs(VmI - VmI1))
        VmIerr = np.sqrt(VmIerr**2 + gerr**2 + ierr**2 + EBVerr**2)
        VmI = VmI + EVI
        if Teff_calib=="C10":
            Teff = C10_Teff_VmI(VmI, MH)
            Teff1 = C10_Teff_VmI(VmI-VmIerr, MH)
            Teff2 = C10_Teff_VmI(VmI+VmIerr, MH)
            Teff_syserr = 59.
        elif Teff_calib=="A99":
            Teff = A99_Teff_VmI(VmI)
            Teff1 = A99_Teff_VmI(VmI-VmIerr)
            Teff2 = A99_Teff_VmI(VmI+VmIerr)
            Teff_syserr = 125.
        color_err = VmIerr
        color = VmI
    if verbose: fp.write("{}={:.2f}±{:.2f}\n".format(Teff_color, color, color_err))
    Teff_err = max(abs(Teff-Teff1), abs(Teff-Teff2))
    if verbose: fp.write("Teff={:.0f} ± {:.0f} (stat) ± {:.0f} (sys)\n".format(Teff,Teff_err,Teff_syserr))
    Teff_err = np.sqrt(Teff_err**2 + Teff_syserr**2)
    
    logg = iterate_find_logg(Teff, logg_mag_dict[logg_mag], MH, dmod, logg_mag)
    try:
        logg = logg[0]
    except:
        pass
    logg_err = phot_logg_error(Teff_err/Teff, dmoderr, magerr=logg_magerr_dict[logg_mag])
    if verbose: fp.write("logg ({})={:.2f} ± {:.2f} (stat)\n".format(logg_mag, logg, logg_err))
    
    vt_syserr = 0.13 # from scatter around B05 relation
    vt = logg_to_vt_B05(logg)
    vt1 = logg_to_vt_B05(logg-logg_err)
    vt2 = logg_to_vt_B05(logg+logg_err)
    vt_err = max(abs(vt-vt1),abs(vt-vt2))
    if verbose: fp.write("vt={:.2f} ± {:.2f} (stat) ± {:.2f} (sys)\n".format(vt, vt_err, vt_syserr))
    vt_err = np.sqrt(vt_syserr**2 + vt_err**2)
    
    if full_output:
        return Teff, Teff_err, logg, logg_err, vt, vt_err, color, color_err, g, r, i, z
    return Teff, Teff_err, logg, logg_err, vt, vt_err

def parsec_des_stellar_params(dmod=0):
    """ Uses label=2 and 3 (subgiant/RGB) to create gmag, rmag->Teff,logg """
    isos = load_parsec_isochrones("DECAM")
    g_Teff_funcs = {}
    g_logg_funcs = {}
    r_Teff_funcs = {}
    r_logg_funcs = {}
    gmr_Teff_funcs = {}
    gmr_logg_funcs = {}
    interp_kwargs = {"bounds_error":False,"fill_value":np.nan}
    for key in isos.keys():
        tab = isos[key]
        tab = tab[(tab["label"]==2) | (tab["label"]==3)]
        gmag, rmag = tab["gmag"], tab["rmag"]
        logT, logg = tab["logTe"], tab["logg"]
        Teff = 10**logT
        g_Teff_funcs[key] = interpolate.interp1d(gmag+dmod,Teff,**interp_kwargs)
        g_logg_funcs[key] = interpolate.interp1d(gmag+dmod,logg,**interp_kwargs)
        r_Teff_funcs[key] = interpolate.interp1d(rmag+dmod,Teff,**interp_kwargs)
        r_logg_funcs[key] = interpolate.interp1d(rmag+dmod,logg,**interp_kwargs)
        gmr_Teff_funcs[key] = interpolate.interp1d(gmag-rmag,Teff,**interp_kwargs)
        gmr_logg_funcs[key] = interpolate.interp1d(gmag-rmag,logg,**interp_kwargs)
    return g_Teff_funcs, g_logg_funcs, r_Teff_funcs, r_logg_funcs, gmr_Teff_funcs, gmr_logg_funcs
def dartmouth_des_stellar_params(dmod=0,ages=[10.0,11.0,12.0,13.0,14.0],logZs=[-2.5,-2.0,-1.5],alpha="ap4"):
    """ Uses label=2 and 3 (subgiant/RGB) to create gmag, rmag->Teff,logg """
    isos = {}
    for MH in [-2.5,-2.0,-1.5]:
        _isos = load_dartmouth_isochrones(MH,alpha,"DECAM")
        for key in _isos.keys():
            tab = _isos[key]
            tab = tab[tab["EEP"] > 111]
            isos[key] = tab
    g_Teff_funcs = {}
    g_logg_funcs = {}
    r_Teff_funcs = {}
    r_logg_funcs = {}
    gmr_Teff_funcs = {}
    gmr_logg_funcs = {}
    interp_kwargs = {"bounds_error":False,"fill_value":np.nan}
    for key in isos.keys():
        tab = isos[key]
        gmag, rmag = tab["gmag"], tab["rmag"]
        logT, logg = tab["logTe"], tab["logg"]
        Teff = 10**logT
        g_Teff_funcs[key] = interpolate.interp1d(gmag+dmod,Teff,**interp_kwargs)
        g_logg_funcs[key] = interpolate.interp1d(gmag+dmod,logg,**interp_kwargs)
        r_Teff_funcs[key] = interpolate.interp1d(rmag+dmod,Teff,**interp_kwargs)
        r_logg_funcs[key] = interpolate.interp1d(rmag+dmod,logg,**interp_kwargs)
        gmr_Teff_funcs[key] = interpolate.interp1d(gmag-rmag,Teff,**interp_kwargs)
        gmr_logg_funcs[key] = interpolate.interp1d(gmag-rmag,logg,**interp_kwargs)
    return g_Teff_funcs, g_logg_funcs, r_Teff_funcs, r_logg_funcs, gmr_Teff_funcs, gmr_logg_funcs

def photometric_stellarparam_derivatives(Teff, logg,
                                         dTdcolor,dvtdlogg=None,
                                         color=None, dTdcolor_func=None):
    """
    Computes dTeff/dlogg, dvt/dlogg assuming purely photometric determinations
    This can be used to get the stellar parameter covariances/correlations.
    
    Input:
      Teff: effective temperature in Kelvin
      logg: surface gravity
      dTdcolor: derivative of effective temperature with respect to color (e.g., g-r)
                Currently you have to compute outside and specify it
      dvtdlogg: derivative of microturbulence with respect to logg
                By default, computes dvt/dlogg using B05 relation. You can specify a number
                here to overwrite the behavior.
    
    Returns:
      dloggdTeff, dvtdlogg
    You can convert these to covariances with these formulas:
      Cov(T,g) = dg/dT * sigma_T^2
      Cov(v,g) = dv/dg * sigma_g^2
      Cov(T,v) = dv/dg * dg/dT * sigma_T^2
    Or correlations:
      Corr(T,g) = dg/dT * sigma_T/sigma_g
      Corr(v,g) = dv/dg * sigma_g/sigma_v
      Corr(T,v) = Corr(T,g) * Corr(v,g)
    """
    
    dloggdT = 4/(np.log(10) * Teff) + 0.4/dTdcolor
    
    if dvtdlogg is None or dvtdlogg=="B05":
        # This is the analytic derivative of the Barklem+2005 relation
        dvtdlogg = 0.173 * logg - 0.6897
    elif dvtdlogg == "M08":
        dvdtdlogg = -0.322
    elif dvtdlogg == "K09":
        dvdtdlogg = -0.23
    return dloggdT, dvtdlogg
