from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from astropy import table
from astropy.io import ascii
from matplotlib.ticker import MultipleLocator

from . import read_data as rd

import seaborn as sns
sns.set(context='talk',style='ticks',font='serif',palette='muted')
sns.set_style({"xtick.direction":"in","ytick.direction":"in"})

######################
# UFD plotting stuff #
######################
_galaxies = ['Bootes I', 'Bootes II', 'CVn II', 'ComBer', 'Hercules', 'Leo IV',
             'Ret II', 'Segue 1', 'Segue 2', 'Tri II', 'Tuc II', 'UMa II',
             'Tuc III', 'Hor I', 'Gru I', 'Psc II', 'Car II', 'Car III', 'Gru II']
_colors = ['#ad8150', #light brown
           (0.86, 0.71786666666666665, 0.33999999999999997),
           (0.65546666666666642, 0.86, 0.33999999999999997),
           (0.33999999999999997, 0.86, 0.37119999999999997),
           (0.33999999999999997, 0.86, 0.71786666666666665),
           (0.33999999999999997, 0.65546666666666642, 0.86),
           "#7b0323", #wine red
           (0.37119999999999997, 0.33999999999999997, 0.86),
           (0.7178666666666661, 0.33999999999999997, 0.86),
           "#8fff9f", #mint green
           '#ffb07c', #peach
           (0.86, 0.33999999999999997, 0.65546666666666642),
           "#02d8e9", #aqua blue
           "#ffb7ce", #baby pink
           '#6e750e', #olive
           '#8e82fe', #periwinkle
           'blue',
           'orange',
           'black' # TODO change this and the marker for Gru II, arbitrary for now to not crash things
          ]
_markers = ['D', '^', '*', 'o', 'H', 's', '*', 'p', '*', '^', 'v', 'h',
            'v', 's', 's','^','o','s','o']
assert len(_galaxies) == len(_colors)
assert len(_galaxies) == len(_markers)
ufd_color = dict(zip(_galaxies, _colors))
ufd_marker= dict(zip(_galaxies, _markers))


#########################
# For plotting patterns #
#########################
def plot_element_labels(ax,Zlist,y_odd,y_even,**kwargs):
    for Z in Zlist:
        elem = species_to_element(int(Z)).split()[0]
        ax.text(Z,y_odd if (Z % 2) else y_even,elem,
                ha='center',va='center',**kwargs)
def plot_elem_series(ax, abunds, offset=0.0, Zmin=None, Zmax=None,
                     errs=None, e_kws={}, **kwargs):
    """
    Plots a single series against atomic number (e.g., abundance patterns or residuals)
    Assumes the index of the series says what element it corresponds to
    """
    import plot_ncap_pattern
    ## Replace with "nice" abundances
    abunds = abunds.copy()
    elems = abunds.index
    elems = [rd.getelem(elem) for elem in elems]
    abunds.index = elems

    Zarr = plot_ncap_pattern.get_Zarr(abunds)
    if Zmin==None: Zmin = min(Zarr)
    if Zmax==None: Zmax = max(Zarr)
    abunds = plot_ncap_pattern.restrict_Z(abunds,Zmin,Zmax)
    Zarr = plot_ncap_pattern.get_Zarr(abunds)

    ax.scatter(Zarr,np.array(abunds) + offset,**kwargs)
    if np.all(errs != None):
        ## Assumes elems and Zarr are the same
        errs = np.array(plot_ncap_pattern.restrict_Z(errs,Zmin,Zmax))
        ax.errorbar(Zarr, np.array(abunds) + offset, yerr=errs, fmt='none', drawstyle='none', **e_kws)

def plot_XH_XH(ax, elem_x, elem_y, data, **kwargs):
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XH', ytype='XH', **kwargs)
def plot_XH_epsX(ax, elem_x, elem_y, data, **kwargs):
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XH', ytype='eps', **kwargs)
def plot_XH_XFe(ax, elem_x, elem_y, data, **kwargs):
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XH', ytype='XFe', **kwargs)
def plot_XFe_XFe(ax, elem_x, elem_y, data, **kwargs):
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XFe', ytype='XFe', **kwargs)
def plot_XFe_XH(ax, elem_x, elem_y, data, **kwargs):
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XFe', ytype='XH', **kwargs)
def plot_XH_AB(ax, elem_x, elem_y, data, **kwargs):
    assert len(elem_y) == 2, elem_y
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XH', ytype='AB', **kwargs)
def plot_XFe_AB(ax, elem_x, elem_y, data, **kwargs):
    assert len(elem_y) == 2, elem_y
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='XFe', ytype='AB', **kwargs)
def plot_AB_AB(ax, elem_x, elem_y, data, **kwargs):
    assert len(elem_x) == 2, elem_x
    assert len(elem_y) == 2, elem_y
    plot_elem_pair(ax, elem_x, elem_y, data, xtype='AB', ytype='AB', **kwargs)

def plot_elem_pair(ax, elem_x, elem_y, data, xtype, ytype,
                   plot_xlimit=False, plot_ylimit=False, plot_xerr=False, plot_yerr=False, 
                   e_kws={}, ulkws={}, **kwargs):
    """
    Plot a 2D scatterplot allowing for errorbars and upper limits

    TODO by default, get color, edgecolor, alpha and apply to error bars/upper limit arrows too

    Parameters
    ----------
    ax : Axes object
        Where to make the plot
    elem_x, elem_y : string
        Elements to plot on x and y axes, put into rd column maker
    data : DataFrame
        Data table with standard column names containing stars to plot (not modified)
        Note if data contains columns with 'ul' (i.e. upper limits), the upper limits
        are automatically removed from plotting unless plot_[x/y]limit is specified.
    xtype, ytype : string
        one of 'eps', 'XH', 'XFe'; determines which of the three abundance types to plot
    plot_xlimit, plot_ylimit : bool
        Whether to draw arrows for limits. data must have 'ul'. Default False.
    plot_xerr, plot_yerr : bool
        Whether to draw errorbars. data must have 'e_'. Default False. 
    e_kws, ulkws : dict
    **kwargs : fed into 

    """
    ## Verify that the data you want to plot is in the data table
    colname_map = {'eps':rd.epscol,'XH':rd.XHcol,'XFe':rd.XFecol,'AB':rd.ABcol}
    assert xtype in colname_map, xtype
    assert ytype in colname_map, ytype
    xcol = colname_map[xtype](elem_x)
    ycol = colname_map[ytype](elem_y)
    assert xcol in data, xcol
    assert ycol in data, ycol
    
    if plot_xerr:
        #assert xtype != "AB", "AB does not support errors"
        if xtype == "AB":
            e_xcol = rd.eABcol(elem_x)
        else:
            e_xcol = rd.errcol(elem_x)
        assert e_xcol in data, e_xcol
    if plot_yerr:
        #assert ytype != "AB", "AB does not support errors"
        if ytype == "AB":
            e_ycol = rd.eABcol(elem_y)
        else:
            e_ycol = rd.errcol(elem_y)
        assert e_ycol in data, e_ycol
    if xtype != "AB":
        ulxcol = rd.ulcol(elem_x)
        if plot_xlimit: assert ulxcol in data, ulxcol
    else:
        ## print("Assuming no upper limits for xtype=AB")
        ulxcol = None
    if ytype != "AB":
        ulycol = rd.ulcol(elem_y)
        if plot_ylimit: assert ulycol in data, ulycol
    else:
        ## print("Assuming no upper limits for ytype=AB")
        ulycol = None
        
    ## Load data to plot
    xplot = np.array(data[xcol].copy())
    yplot = np.array(data[ycol].copy())
    if plot_xerr: xerr = np.array(data[e_xcol].copy())
    else: xerr = None
    if plot_yerr: yerr = np.array(data[e_ycol].copy())
    else: yerr = None
        
    ## Filter out upper limits, if plotting limits save those values
    if ulxcol in data:
        upper_limits_x = np.array(data[ulxcol],dtype=bool)
        if np.sum(upper_limits_x) > 0:
            if plot_xlimit: # Save the limit locations before removing
                xlimit_x = xplot[upper_limits_x].copy()
                xlimit_y = yplot[upper_limits_x].copy()
        else: plot_xlimit = False # No x limits to plot
    if ulycol in data:
        upper_limits_y = np.array(data[ulycol],dtype=bool)
        if np.sum(upper_limits_y) > 0:
            if plot_ylimit: # Save the limit locations before removing
                ylimit_x = xplot[upper_limits_y].copy()
                ylimit_y = yplot[upper_limits_y].copy()
        else: plot_ylimit = False # No y limits to plot
    if ulxcol in data:
        # Remove data with x limits only after saving both x and y
        xplot[upper_limits_x] = np.nan
        yplot[upper_limits_x] = np.nan
    if ulycol in data:
        # Remove data with y limits
        xplot[upper_limits_y] = np.nan
        yplot[upper_limits_y] = np.nan
    ## Plot actual data
    ax.scatter(xplot,yplot,**kwargs)
    ## Plot error bars if requested
    if plot_xerr or plot_yerr:
        # TODO make smart defaults about e_kws based on kwargs if not specified
        ax.errorbar(xplot,yplot,xerr=xerr,yerr=yerr,fmt='none',**e_kws)
    ## Plot upper limits if requested
    if xtype != "AB" and plot_xlimit:
        plot_limits(ax,xlimit_x,xlimit_y,direction='-x',**ulkws)
    if ytype != "AB" and plot_ylimit:
        plot_limits(ax,ylimit_x,ylimit_y,direction='-y',**ulkws)
    ## Plot AB limits
    if xtype == "AB" and plot_xlimit:
        A, B = xcol[1:-1].split("/")
        Acol, Bcol = rd.XHcol(A), rd.XHcol(B)
        Aplot, Bplot = np.array(data[Acol].copy()), np.array(data[Bcol].copy())
        ABplot = Aplot - Bplot
        ulAcol, ulBcol = rd.ulcol(A), rd.ulcol(B)
        ulA, ulB = np.array(data[ulAcol],dtype=bool), np.array(data[ulBcol],dtype=bool)
        ## Upper limits
        ii = ulA & (~ulB)
        plot_limits(ax, ABplot[ii], yplot[ii], direction="-x", **ulkws)
        ## Lower limits
        ii = (~ulA) & ulB
        plot_limits(ax, ABplot[ii], yplot[ii], direction="+x", **ulkws)
    if ytype == "AB" and plot_ylimit:
        A, B = ycol[1:-1].split("/")
        Acol, Bcol = rd.XHcol(A), rd.XHcol(B)
        Aplot, Bplot = np.array(data[Acol].copy()), np.array(data[Bcol].copy())
        ABplot = Aplot - Bplot
        ulAcol, ulBcol = rd.ulcol(A), rd.ulcol(B)
        ulA, ulB = np.array(data[ulAcol],dtype=bool), np.array(data[ulBcol],dtype=bool)
        ## Upper limits
        ii = ulA & (~ulB)
        plot_limits(ax, xplot[ii], ABplot[ii], direction="-y", **ulkws)
        ## Lower limits
        ii = (~ulA) & ulB
        plot_limits(ax, xplot[ii], ABplot[ii], direction="+y", **ulkws)
    

def plot_limits(ax,xplot,yplot,direction='-y',arrow_length=0.4,scatter_kws={},arrow_kws={}):
    assert len(xplot)==len(yplot)
    assert arrow_length > 0
    ## Identify direction; 
    direction_map = {'+x': ( 1, 0), '-x': (-1, 0),
                     '+y': ( 0, 1), '-y': ( 0,-1)}
    dx,dy = direction_map[direction]
    dx *= arrow_length; dy *= arrow_length
    for x,y in zip(xplot,yplot):
        # TODO make smart defaults about scatter_kws based on kwargs
        ax.scatter(x,y,**scatter_kws)
        ax.arrow(x,y,dx,dy,**arrow_kws)

def plot_many_ufds_FeH_XFe(ax,elem,ufds,skipgals=[],ufdsize=60,fill_scatter=False):
    _plot_many_ufds_FeH(plot_XH_XFe,ax,elem,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter)
def plot_many_ufds_FeH_XH(ax,elem,ufds,skipgals=[],ufdsize=60,fill_scatter=False):
    _plot_many_ufds_FeH(plot_XH_XH,ax,elem,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter)

def plot_many_ufds_XH_XH(ax,elem1,elem2,ufds,skipgals=[],ufdsize=60,fill_scatter=False,
                         plot_xlimit=False, plot_ylimit=False):
    _plot_many_ufds(plot_XH_XH,ax,elem1,elem2,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter,
                    plot_xlimit=plot_xlimit, plot_ylimit=plot_ylimit)
def plot_many_ufds_XH_XFe(ax,elem1,elem2,ufds,skipgals=[],ufdsize=60,fill_scatter=False,
                           plot_xlimit=False, plot_ylimit=False):
    _plot_many_ufds(plot_XH_XFe,ax,elem1,elem2,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter,
                    plot_xlimit=plot_xlimit, plot_ylimit=plot_ylimit)
def plot_many_ufds_XFe_XFe(ax,elem1,elem2,ufds,skipgals=[],ufdsize=60,fill_scatter=False,
                           plot_xlimit=False, plot_ylimit=False):
    _plot_many_ufds(plot_XFe_XFe,ax,elem1,elem2,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter,
                    plot_xlimit=plot_xlimit, plot_ylimit=plot_ylimit)
def plot_many_ufds_XFe_AB(ax,elem1,elem2,ufds,skipgals=[],ufdsize=60,fill_scatter=False):
    assert len(elem2)==2, elem2
    _plot_many_ufds(plot_XFe_AB,ax,elem1,elem2,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter)
def plot_many_ufds_XH_AB(ax,elem1,elem2,ufds,skipgals=[],ufdsize=60,fill_scatter=False):
    assert len(elem2)==2, elem2
    _plot_many_ufds(plot_XH_AB,ax,elem1,elem2,ufds,skipgals=skipgals,ufdsize=ufdsize,fill_scatter=fill_scatter)
def get_ufd_colors_markers(ufds):
    colors = [ufd_color[gal] for gal in np.unique(ufds['galaxy'])]
    markers = [ufd_marker[gal] for gal in np.unique(ufds['galaxy'])]
    return colors, markers
def _plot_many_ufds(pltfn,ax,elem1,elem2,ufds,skipgals=[],ufdsize=60,fill_scatter=False,
                    plot_xlimit=False, plot_ylimit=False):
    all_gals = np.unique(ufds['galaxy'])
    colors, markers = get_ufd_colors_markers(ufds)
    assert len(all_gals) == len(colors)
    assert len(all_gals) == len(markers)
    groups = ufds.groupby('galaxy')
    for gal,color,marker in zip(all_gals,colors,markers):
        if gal in skipgals: continue
        this_ufd = groups.get_group(gal)
        if marker=='D': _ufdsize = ufdsize*.5
        else: _ufdsize = ufdsize
        ufd_kws = construct_dictionaries(color,marker,_ufdsize,fill_scatter=fill_scatter)
        if len(this_ufd)==3:
            ufd_kws['color'] = [color,color,color]
        pltfn(ax,elem1,elem2,this_ufd,plot_xlimit=plot_xlimit,plot_ylimit=plot_ylimit,
              label=gal,**ufd_kws)
def _plot_many_ufds_FeH(pltfn,ax,elem,ufds,skipgals=[],ufdsize=60,fill_scatter=False,plot_ylimit=True):
    all_gals = np.unique(ufds['galaxy'])
    colors, markers = get_ufd_colors_markers(ufds)
    
    if rd.XFecol(elem) not in ufds: return
    groups = ufds.groupby('galaxy')
    for gal,color,marker in zip(all_gals,colors,markers):
        if gal in skipgals: continue
        this_ufd = groups.get_group(gal)
        if marker=='D': _ufdsize = ufdsize*.5
        else: _ufdsize = ufdsize
        ufd_kws = construct_dictionaries(color,marker,_ufdsize,fill_scatter=fill_scatter)
        if len(this_ufd)==3:
            ufd_kws['color'] = [color,color,color]
        pltfn(ax,'Fe',elem,this_ufd,plot_ylimit=plot_ylimit,
              label=gal,**ufd_kws)
def construct_dictionaries(color,marker,size,
                           scatter_ecolor='k',
                           scatter_elinewidth=1.5,
                           alpha=1.0,
                           fill_scatter=False,
                           elinewidth=1,capsize=0,
                           arrow_head_length=0.15,
                           arrow_head_width=0.05,
                           arrow_length=0.25):
    """
    Example usage:
    halo_kws = construct_dictionaries('k','o', 20, alpha=.3)
    pltabund.plot_XFe_XFe(ax, 'K', 'Mg', roed, plot_xlimit=True, plot_ylimit=True, label="Halo",
                          **halo_kws)
    """
    e_kws = {'ecolor':color,'elinewidth':elinewidth,'capsize':capsize}
    if fill_scatter:
        ulkws = {'arrow_length':arrow_length,
                 'scatter_kws':{'marker':marker,'s':size,'facecolor':color,'alpha':alpha},
                 'arrow_kws':{'color':color,'head_length':arrow_head_length,'head_width':arrow_head_width,'alpha':alpha}
                 }
    else:
        ulkws = {'arrow_length':arrow_length,
                 'scatter_kws':{'marker':marker,'s':size,'facecolor':'none',
                                'linewidths':1,'edgecolors':color,'alpha':alpha},
                 'arrow_kws':{'color':color,'head_length':arrow_head_length,'head_width':arrow_head_width,'alpha':alpha}
                 }
    kws = {'color':color,'edgecolors':scatter_ecolor,'linewidths':scatter_elinewidth,'marker':marker,'s':size,'alpha':alpha,
           'e_kws':e_kws,'ulkws':ulkws}
    return kws
def add_ufd_legend(fig, ax, hbuffer=.09, height=.09, nrow=3, lax=None,
                   fontsize=12, loc='center', **kwargs):
    handles,labels = ax.get_legend_handles_labels()
    for handle,label in zip(handles,labels):
        try:
            if label=='CVn II': handle.set_sizes([60])
        except:
            continue
                
    if lax is None:
        lax = fig.add_axes([hbuffer,0,1.-2*hbuffer,height],frameon=False)
    ncol = int(len(handles)/nrow)#+(len(handles) % nrow)
    if len(handles) > nrow*ncol: ncol += 1
    lax.legend(handles,labels,fontsize=fontsize,scatterpoints=1,ncol=ncol,loc=loc, **kwargs)
    lax.set_xticks([])
    lax.set_yticks([])

def confidence_ellipse(ax, x, y, var_x, var_y, cov_xy, n_std=1.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse
    given variance of x, y, and their covariance

    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    x, y : float
        where to center the ellipse

    var_x, var_y, cov_xy : float
        parameters of the ellipse

    n_std : float (default 1.0)
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    pearson = cov_xy/np.sqrt(var_x * var_y)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      **kwargs)
    scale_x = np.sqrt(var_x) * n_std
    scale_y = np.sqrt(var_y) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

