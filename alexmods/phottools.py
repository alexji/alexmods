# coding: utf-8
""" Some photometry tools for stellar spectroscopists """
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from scipy import interpolate

import logging
import os, sys, time
logger = logging.getLogger(__name__)

__all__ = []

from .read_data import datapath


def eval_BC(Teff,logg,FeH,filt="g",allBCs=None):
    """
    Default is alpha/Fe = +0.4
    """
    if allBCs is None: allBCs = read_bc_table()
        
    BCs = allBCs[filt]
    
    points = np.atleast_2d([Teff,logg,FeH]).T
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
    assert np.all(np.ravel(gmi) < 2.1)
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

##################################################################
# From Casagrande et al. 2010
##################################################################
def C10_Teff_BmV(BmV, FeH):
    #73K scatter
    a0, a1, a2, a3, a4, a5 = .5665, .4809, -.0060, -.0613, -.0042, -.0055
    theta = a0 + a1*BmV + a2*BmV*BmV + a3*BmV*FeH + a4*FeH + a5*FeH*FeH
    Teff = 5040./theta
    return Teff
def C10_Teff_VmI(VmI, FeH):
    #59K scatter
    a0, a1, a2, a3, a4, a5 = .4033, .8171, -.1987, -.0409, .0319, .0012
    theta = a0 + a1*VmI + a2*VmI*VmI + a3*VmI*FeH + a4*FeH + a5*FeH*FeH
    Teff = 5040./theta
    return Teff


def phot_logg(Teff,mag0,BCmag,distmod,Mstar=0.8):
    """
    Using solar values from Venn et al. 2017
    """
    return 4.44 + np.log10(Mstar) + 4*np.log10(Teff/5780) + 0.4 * (mag0 - distmod + BCmag - 4.75)

