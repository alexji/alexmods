#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An object for dealing with one-dimensional spectra. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from six import string_types

__all__ = []

## Standard imports
import logging
import numpy as np
from scipy import interpolate, signal
import os, sys, time
from collections import OrderedDict
import random

## Astro imports
from astropy.io import fits, ascii
from alexmods.specutils.spectrum import (Spectrum1D, read_mike_spectrum, stitch as spectrum_stitch)

## GUI imports
# https://pythonspot.com/pyqt5-matplotlib/
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox
from PyQt5.QtWidgets import QWidget, QPushButton, QDialog, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from .mpl import MPLWidget


DOUBLE_CLICK_INTERVAL = 0.1 # MAGIC HACK

def median_filter(spec, window):
    newflux = signal.median_filter(spec.flux, window)
    return Spectrum1D(spec.dispersion, newflux, spec.ivar, spec.metadata)

def fit_continuum_lsq(spec, knots, exclude=[], maxiter=3, sigma_lo=2, sigma_hi=2,
                      get_edges=False, **kwargs):
    """ Fit least squares continuum through spectrum data using specified knots, return model """
    assert np.all(np.array(list(map(len, exclude))) == 2), exclude
    assert np.all(np.array(list(map(lambda x: x[0] < x[1], exclude)))), exclude
    x, y, w = spec.dispersion, spec.flux, spec.ivar
    
    # This is a mask marking good pixels
    mask = np.ones_like(x, dtype=bool)
    # Exclude regions
    for xmin, xmax in exclude:
        mask[(x >= xmin) & (x <= xmax)] = False
    # Get rid of bad fluxes
    mask[np.abs(y)<1e-6] = False
    mask[np.isnan(y)] = False
    if get_edges:
        left = np.where(mask)[0][0]
        right = np.where(mask)[0][-1]
    
    for iter in range(maxiter):
        # Make sure there the knots don't hit the edges
        wmin = x[mask].min()
        wmax = x[mask].max()
        while knots[-1] >= wmax: knots = knots[:-1]
        while knots[0] <= wmin: knots = knots[1:]
        
        try:
            fcont = interpolate.LSQUnivariateSpline(x[mask], y[mask], knots, w=w[mask], **kwargs)
        except ValueError:
            print("Knots:",knots)
            print("xmin, xmax = {:.4f}, {:.4f}".format(wmin, wmax))
            raise
        # Iterative rejection
        cont = fcont(x)
        sig = (cont-y) * np.sqrt(w)
        sig /= np.nanstd(sig)
        mask[sig > sigma_hi] = False
        mask[sig < -sigma_lo] = False
    if get_edges:
        return fcont, left, right
    return fcont

def load_spectra_by_order(fnames, fluxband=7):
    """
    Load data, sort by order number
    """
    all_orders = OrderedDict()
    for fname in fnames:
        specs = read_mike_spectrum(fname,fluxband=fluxband)
        for iord, spec in specs.items():
            basename = os.path.basename(fname)
            spec.metadata["filename"] = basename
            if iord in all_orders:
                all_orders[iord][basename] = spec
            else:
                all_orders[iord] = OrderedDict()
                all_orders[iord][basename] = spec
    sorted_all_orders = OrderedDict(sorted(all_orders.items(), key=lambda x: x[0]))
    return sorted_all_orders

def initialize_knots(wmin, wmax, knot_spacing):
    """ Place knots evenly through """
    waverange = wmax - wmin - 2*knot_spacing
    Nknots = int(waverange // knot_spacing)
    minknot = wmin + (waverange - Nknots * knot_spacing)/2.
    xknots = np.arange(minknot, wmax, knot_spacing)
    # Make sure there the knots don't hit the edges
    while xknots[-1] >= wmax - knot_spacing: xknots = xknots[:-1]
    while xknots[0] <= wmin + knot_spacing: xknots = xknots[1:]
    return list(xknots)
def initialize_knots_from_spectrum(spec, knot_spacing):
    wave = spec.dispersion
    wmin, wmax = wave.min(), wave.max()
    return initialize_knots(wmin, wmax, knot_spacing)
def initialize_knots_from_spectra(specs, knot_spacing):
    wmin = -np.inf
    wmax = np.inf
    for label, spec in specs.items():
        wave = spec.dispersion
        wmin = max(wave.min(), wmin)
        wmax = min(wave.max(), wmax)
    return initialize_knots(wmin, wmax, knot_spacing)

def fit_continuum_to_spectra(specs, knots, sigma_lo_dict, sigma_hi_dict,
                             snip_edges=True, **kwargs):
    """ Run lsq spline fit on all spectra """
    cont_funcs = OrderedDict()
    cont_data = OrderedDict()
    for label, spec in specs.items():
        sigma_lo, sigma_hi = sigma_lo_dict[label], sigma_hi_dict[label]
        fcont, left, right = fit_continuum_lsq(spec, knots, get_edges=True,
                                               sigma_lo=sigma_lo, sigma_hi=sigma_hi,
                                               **kwargs)
        cont_funcs[label] = fcont
        dcont = fcont(spec.dispersion)
        if snip_edges:
            # Set continuum = nan at edges
            dcont[:left] = np.nan
            dcont[right:] = np.nan
        cont_data[label] = dcont
    return cont_funcs, cont_data

class ContinuumModel(object):
    def __init__(self, degree=3, knot_spacing=10., sigma_lo=2, sigma_hi=2):
        self.degree = degree
        self.knot_spacing = knot_spacing
        self.sigma_lo_default = sigma_lo
        self.sigma_hi_default = sigma_hi
        self._initialize()
    
    def _initialize(self):
        ## Data to fit
        self.all_specs = OrderedDict()
        self.all_order_numbers = []
        
        ## Data for each order
        self.all_knots = OrderedDict()
        self.all_exclude_regions = OrderedDict()
        
        ## Data for each spectrum of each order
        self.all_y_knots = OrderedDict()
        self.all_continuum_data = OrderedDict()
        self.all_continuum_functions = OrderedDict()
        
        self.label_to_rv = {}
        self.rv_applied = False
        
        # order -> sigma_lo/hi_dict
        self.all_sigma_lo = OrderedDict()
        self.all_sigma_hi = OrderedDict()

    def load_data(self, fnames, labels=None, fluxband=7,
                  label_to_rv=None):
        """
        Load data, sort by order number
        """
        self.fluxband=fluxband
        if isinstance(fnames, string_types):
            fnames = [fnames]
        if labels is not None:
            if isinstance(labels, string_types):
                labels = [labels]
            assert len(fnames) == len(labels), (fnames, labels)
        else:
            labels = [os.path.basename(fname) for fname in fnames]
        
        # Create empty structures
        self._initialize()
        
        # Read data and fill skeleton structure
        for fname, label in zip(fnames, labels):
            specs = read_mike_spectrum(fname,fluxband=fluxband)
            for iord, spec in specs.items():
                spec.metadata["filename"] = label
                if iord not in self.all_specs:
                    self.all_specs[iord] = OrderedDict()
                    self.all_continuum_data[iord] = OrderedDict()
                    self.all_continuum_functions[iord] = OrderedDict()
                    self.all_y_knots[iord] = OrderedDict()
                    
                    self.all_knots[iord] = []
                    self.all_exclude_regions[iord] = []

                    self.all_sigma_hi[iord] = OrderedDict()
                    self.all_sigma_lo[iord] = OrderedDict()
                self.all_specs[iord][label] = spec
                self.all_continuum_data[iord][label] = None
                self.all_continuum_functions[iord][label] = None
                self.all_sigma_hi[iord][label] = self.sigma_hi_default
                self.all_sigma_lo[iord][label] = self.sigma_lo_default
        self.all_order_numbers = list(np.sort(list(self.all_specs.keys())))
        
        self.fnames = [os.path.abspath(fname) for fname in fnames]
        self.labels = labels
        
        if label_to_rv is not None:
            self.apply_radial_velocity_corrections(label_to_rv)
        return
    
    def apply_radial_velocity_corrections(self, label_to_rv):
        ## Verification that all the RVs exist
        for order, specs in self.all_specs.items():
            for label, spec in specs.items():
                assert label in label_to_rv, (order, label)
        for order, specs in self.all_specs.items():
            for label, spec in specs.items():
                rv = label_to_rv[label]
                spec.redshift(rv)
        self.label_to_rv = label_to_rv
        self.rv_applied = True
    def fit_all_continuums(self):
        """ Fit all data for all orders """
        for order in self.all_order_numbers:
            self.fit_continuums(order)
        return
    def fit_continuums(self, order):
        """ Fit all data for one order """
        specs = self.all_specs[order]
        knots = self.get_knots(order)
        exclude = self.all_exclude_regions[order]
        fconts, dconts = fit_continuum_to_spectra(specs, knots,
                                                  self.all_sigma_lo[order],
                                                  self.all_sigma_hi[order],
                                                  k=self.degree, exclude=exclude,
                                                  )
        self.all_continuum_functions[order] = fconts
        self.all_continuum_data[order] = dconts
        # Update y_knots
        for label, spec in specs.items():
            cont = dconts[label]
            ixknots = np.searchsorted(spec.dispersion, knots)
            self.all_y_knots[order][label] = cont[ixknots]
    def get_knots(self, order):
        """ Get knots for this order. Initialize if needed. """
        assert order in self.all_order_numbers, all_order_numbers
        knots = self.all_knots[order]
        if len(knots) == 0:
            specs = self.all_specs[order]
            knots = initialize_knots_from_spectra(specs, self.knot_spacing)
            self.all_knots[order] = knots
        return knots
    def set_knots(self, order, knots):
        self.all_knots[order] = knots
        self.fit_continuums(order)
    def reset_knots(self, order):
        self.set_knots(order, [])
    def add_knot(self, order, x):
        knots = self.all_knots[order]
        ix = np.searchsorted(knots, x)
        knots.insert(ix, x)
        assert np.allclose(np.sort(knots), np.array(knots))
        self.all_knots[order] = knots
        self.fit_continuums(order)
    def delete_knot(self, order, iknot):
        knots = self.all_knots[order]
        x = knots.pop(iknot)
        self.fit_continuums(order)
        return x
    def get_nearest_knot(self, order, x, gety=False):
        knots = self.all_knots[order]
        if len(knots) == 0:
            if gety: return None, None, None
            return None, None
        iknot = np.argmin(np.abs(x-np.array(knots)))
        xknot = knots[iknot]
        if gety:
            yknots = [self.all_y_knots[order][label][iknot] for label in self.labels]
            return iknot, xknot, yknots
        return iknot, xknot
    def delete_nearest_knot(self, order, x):
        iknot, xknot = self.get_nearest_knot(order, x)
        self.delete_knot(order, iknot)
    def add_exclude_region(self, order, xmin, xmax):
        assert xmin < xmax
        self.all_exclude_regions[order].append((xmin,xmax))
    def remove_exclude_region(self, order, iexclude):
        return self.all_exclude_regions[order].pop(iexclude)
    def reset_exclude_regions(self, order):
        self.all_exclude_regions[order] = []
    
    def get_order_labels(self, order):
        """ Get the spectrum labels associated with this order """
        specs = self.all_specs[order]
        return list(specs.keys())
    
    def get_sigma_lohi(self, order, label):
        return self.all_sigma_lo[order][label], self.all_sigma_hi[order][label]
    def set_sigma_lo(self, order, label, sigma):
        self.all_sigma_lo[order][label] = sigma
    def set_sigma_hi(self, order, label, sigma):
        self.all_sigma_hi[order][label] = sigma

    def save(self, fname):
        """
        Save the parameters (degree, knots, exclusion regions) 
        and filenames, but only the location of the original data filenames
        """
        output = [self.fnames, self.labels, self.fluxband,
                  self.degree, self.knot_spacing, self.all_sigma_lo, self.all_sigma_hi,
                  self.all_knots, self.all_exclude_regions,
                  self.label_to_rv, self.rv_applied]
        np.save(fname, output)
        
    @classmethod
    def load(cls, fname):
        """
        Load the data and refit
        """
        model = cls()
        fnames, labels, fluxband, \
            degree, knot_spacing, all_sigma_lo, all_sigma_hi, \
            all_knots, all_exclude_regions, \
            label_to_rv, rv_applied = \
            np.load(fname, allow_pickle=True)
        model.degree = degree
        model.knot_spacing = knot_spacing
        
        print(fnames,labels,fluxband)
        model.load_data(fnames, labels, fluxband,
                        label_to_rv = label_to_rv)
        model.all_sigma_lo = all_sigma_lo
        model.all_sigma_hi = all_sigma_hi
        model.all_knots = all_knots
        model.all_exclude_regions = all_exclude_regions
        model.fit_all_continuums()
        return model

    def load_parameters_from(self, fname):
        """
        Load parameters but keep current data/rv.
        Parameters are:
          degree, knot_spacing, sigma_lo, sigma_hi,
          all_knots, all_exclude_regions
          
        Currently does not do any checks, but in the future would be nice if
        e.g. check that all the orders are there,
        or that orders roughly match up in wavelength.
        """
        data = np.load(fname, allow_pickle=True)
        assert data[2] == self.fluxband, (data[2], self.fluxband)
        # TODO this will fail with the new dictionary stuff!
        self.degree, self.knot_spacing, \
            self.all_sigma_lo, self.all_sigma_hi = data[3:7]
        self.all_knots = data[7]
        self.all_exclude_regions = data[8]

    def get_normalized_orders(self):
        self.fit_all_continuums()
        normalized_specs = OrderedDict()
        for order, specs in self.all_specs.items():
            conts = self.all_continuum_data[order]
            normalized_specs[order] = OrderedDict()
            for label, spec in specs.items():
                cont = conts[label]
                # TODO add some extra metadata?
                meta = spec.metadata.copy()
                badmask = np.logical_or(np.isnan(cont), np.isnan(spec.flux))
                flux = spec.flux/cont
                flux[badmask] = np.nan
                ivar = spec.ivar*cont*cont
                ivar[badmask] = 0.
                norm = Spectrum1D(spec.dispersion, flux, ivar, meta)
                normalized_specs[order][label] = norm
        return normalized_specs
    def get_coadded_orders(self, mode="wavg", **kwargs):
        """
        wavg = weighted average
        sfit = spline fit
        """
        assert mode in ["wavg"]
        normalized_specs = self.get_normalized_orders()
        coadded_specs = OrderedDict()
        for order, specs in normalized_specs.items():
            ordwaves = []
            ordfluxs = []
            ordivars = []
            for label, spec in specs.items():
                ordwaves.append(spec.dispersion)
                ordfluxs.append(spec.flux)
                ordivars.append(spec.ivar)
                dwave = np.median(np.diff(spec.dispersion))
            
            ordwaves = np.vstack(ordwaves)
            ordfluxs = np.vstack(ordfluxs)
            ordivars = np.vstack(ordivars)
            wmin = np.max(ordwaves[:,0])
            wmax = np.min(ordwaves[:,-1])
            Npix = ordwaves.shape[1]
            knots = np.linspace(wmin, wmax, int(Npix//2)+2)[1:-1]
            waveeval = np.linspace(wmin, wmax, Npix)
            for i in range(len(ordwaves)):
                ordfluxs[i] = np.interp(waveeval, ordwaves[i], ordfluxs[i])
                ordivars[i] = np.interp(waveeval, ordwaves[i], ordivars[i])
    
            ordwave = waveeval
            ordivar = np.nansum(ordivars, axis=0)
            ordflux = np.nansum(ordfluxs*ordivars, axis=0)/ordivar
            #ord_meansquare = np.nansum((ordfluxs-ordflux[np.newaxis,:])**2 * ordivars, axis=0)/ordivar
            #ordivar = 1/ord_meansquare
            ordivar[(np.isnan(ordflux)) | np.isnan(ordivar)] = 0.
            #meta = specs[0].metadata.copy()
            meta = {"labels":",".join(list(specs.keys()))}
            coadded_specs[order] = Spectrum1D(waveeval, ordflux, ordivar, meta)
        return coadded_specs
    def get_full_stitch(self):
        coadded_specs = list(self.get_coadded_orders().values())
        return spectrum_stitch(coadded_specs)

def plot_normalized_order_regions(model, xlims, ylims, **kwargs):
    assert len(xlims)==len(ylims)
    def in_xlim_range(spec, xlim):
        for x in xlim:
            if spec.dispersion[0] < x and x < spec.dispersion[-1]: return True
        return False
    N = len(xlims)
    figx, figy = 10, 4
    fig, axes = plt.subplots(N, figsize=(figx, N*figy))
    normalized_specs = model.get_normalized_orders()
    full_stitch = model.get_full_stitch()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colordict = {}
    for i,label in enumerate(model.labels):
        colordict[label] = colors[i % len(colors)]

    for xlim, ylim, ax in zip(xlims, ylims, axes):
        for order, specs in normalized_specs.items():
            for label, spec in specs.items():
                if in_xlim_range(spec, xlim):
                    print("Found one",order,label,xlim)
                    ax.plot(spec.dispersion, spec.flux, 
                            label=label, color=colordict[label],
                            **kwargs)
        ii = (full_stitch.dispersion > xlim[0]-10) & (full_stitch.dispersion < xlim[1]+10)
        ax.plot(full_stitch.dispersion[ii], full_stitch.flux[ii], 'k-', lw=3, alpha=.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    fig.tight_layout()
    return fig
def plot_all_normalized_orders(model, Nrow, Ncol, **kwargs):
    if Nrow*Ncol != len(model.all_order_numbers):
        print("warning: not plotting all orders")
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colordict = {}
    for i,label in enumerate(model.labels):
        colordict[label] = colors[i % len(colors)]
    
    normalized_orders = model.get_normalized_orders()
    
    figx, figy = 6, 3
    fig, axes = plt.subplots(Nrow, Ncol, figsize=(figx*Ncol, figy*Nrow))
    for ax, order in zip(axes.flat, model.all_order_numbers[::-1]):
        specs = normalized_orders[order]
        for label, spec in specs.items():
            ax.plot(spec.dispersion, spec.flux, label=label, color = colordict[label],
                    **kwargs)
        ax.set_ylim(0,1.2)
        ax.set_xlim(get_ax_xlim(ax))
    fig.tight_layout()
    return fig

class NormalizedPlotDialog(QDialog):
    def __init__(self, parent):
        # Center it on the parent location
        rect = parent.geometry()
        x, y = rect.center().x(), rect.center().y()
        w = 1000
        h = 640
        self.setGeometry(x-w/2, y-h/2, w, h)
        vbox = QtGui.QVBoxLayout(self)
        figure_widget = MPLWidget(self, toolbar=True, tight_layout=True)
        vbox.addWidget(figure_widget)
    def make_plot(self, fullstitch, coadds):
        pass

class ContinuumNormalizationApp(QMainWindow):
    def __init__(self, model, 
                 default_save="continuum_fit.npy",
                 default_stitched="stitched_spectrum.txt"):
        super().__init__()

        self.initUI()
        self.default_save = default_save
        self.stitched_default_fname = default_stitched
        self.new_model(model)
        
    def new_model(self, model):
        self.model = model

        # Sorry the naming here is a legacy code problem 
        self.all_orders = self.model.all_specs
        self.all_order_numbers = self.model.all_order_numbers
        
        # Setup label colors
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.colordict = {}
        for i,label in enumerate(self.model.labels):
            self.colordict[label] = colors[i % len(colors)]
        
        self.all_knots = self.model.all_knots
        self.all_y_knots = self.model.all_y_knots
        self.all_continuum_data = self.model.all_continuum_data
        self.all_continuum_functions = self.model.all_continuum_functions
        self.all_exclude_regions = self.model.all_exclude_regions
        
        
        self.set_order(np.max(self.all_order_numbers))
        
    def fit_all_continuums(self):
        self.model.fit_all_continuums()

    def set_order(self, order):
        """ Set which order to plot """
        assert order in self.all_order_numbers
        self.order = order
        self.labels = self.model.get_order_labels(order)
        self.set_label(order, self.labels[0])
        self.canvas.set_title("Order {}".format(self.order))
        self.fit_continuums()
        self.update_plot()
    def set_order_gui(self):
        """ Set which order to plot """
        order = int(self.ledit_change_order.text())
        self.set_order(order)
    def set_next_label(self):
        ix = self.labels.index(self.label)
        self.set_label(self.order, self.labels[(ix+1) % len(self.labels)])
    def set_prev_label(self):
        ix = self.labels.index(self.label)
        self.set_label(self.order, self.labels[(ix-1) % len(self.labels)])
    def set_label(self, order, label):
        ## Update label
        self.label = label
        self.label_label.setText(label)
        ## Update sigma_lo/hi
        slo, shi = self.model.get_sigma_lohi(order, label)
        self.ledit_siglo.setText(str(slo))
        self.ledit_sighi.setText(str(shi))
        return
    def set_sigma_lo(self, siglo=None):
        siglo = siglo or float(self.ledit_siglo.text())
        assert siglo >= 0, siglo
        self.model.set_sigma_lo(self.order, self.label, siglo)
        self.fit_continuums()
        self.update_plot(False)
        self.ledit_siglo.setText(str(siglo))
    def set_sigma_hi(self, sighi=None):
        sighi = sighi or float(self.ledit_sighi.text())
        assert sighi >= 0, sighi
        self.model.set_sigma_hi(self.order, self.label, sighi)
        self.fit_continuums()
        self.update_plot(False)
        self.ledit_sighi.setText(str(sighi))
    def increment_sigma_lo(self):
        siglo, _ = self.model.get_sigma_lohi(self.order, self.label)
        self.set_sigma_lo(round(siglo+0.1,1))
    def decrement_sigma_lo(self):
        siglo, _ = self.model.get_sigma_lohi(self.order, self.label)
        self.set_sigma_lo(round(siglo-0.1,1))
    def increment_sigma_hi(self):
        _, sighi = self.model.get_sigma_lohi(self.order, self.label)
        self.set_sigma_hi(round(sighi+0.1,1))
    def decrement_sigma_hi(self):
        _, sighi = self.model.get_sigma_lohi(self.order, self.label)
        self.set_sigma_hi(round(sighi-0.1,1))
    def apply_sigma_to_order(self):
        siglo = float(self.ledit_siglo.text())
        sighi = float(self.ledit_sighi.text())
        for label in self.labels:
            self.model.set_sigma_lo(self.order, label, siglo)
            self.model.set_sigma_hi(self.order, label, sighi)
        self.fit_continuums()
        self.update_plot(False)
    
    def update_plot(self, reset_limits=True):
        self.canvas.textinfo("")
        self.plot_data(reset_limits=reset_limits)
        self.canvas2.axhline(1,color='k',ls=':',lw=1, zorder=-9)
        self.plot_continuums()
        self.plot_exclude_regions()
        self.canvas.draw()
        self.canvas2.draw()
    def plot_data(self, reset_limits=True):
        self.canvas.clear_plot()
        self.canvas2.clear_plot()
        specs = self.all_orders[self.order]
        for label, spec in specs.items():
            self.canvas.plot(spec.dispersion, spec.flux, ',-', label=label, lw=.2,
                             color=self.colordict[label])
        if reset_limits:
            self.canvas.reset_limits()
            self.canvas2.set_xlim(self.canvas.get_xlim())
            self.canvas2.set_ylim(0,1.2)
        self.canvas.legend()
    def plot_continuums(self):
        try:
            specs = self.all_orders[self.order]
            conts = self.all_continuum_data[self.order]
            knots = self.all_knots[self.order]
            y_knots = self.all_y_knots[self.order]
            for label, spec in specs.items():
                cont = conts[label]
                self.canvas.plot(spec.dispersion, cont, '-', lw=1.5, alpha=.5, color=self.colordict[label])
                yknots = y_knots[label]
                self.canvas.plot(knots, yknots, 'o', color=self.colordict[label])
                # Plot normalized data
                self.canvas2.plot(spec.dispersion, spec.flux/cont, '-', lw=.5, alpha=.5, color=self.colordict[label])
        except:
            print("Error plotting continuum {}".format(self.order))
    def plot_exclude_regions(self):
        exclude = self.all_exclude_regions[self.order]
        for xmin, xmax in exclude:
            self.canvas.vspan(xmin, xmax)
    
    def fit_continuums(self):
        self.model.fit_continuums(self.order)
        
    def initUI(self):
        self.title = "Continuum Normalization"
        self.left = 10
        self.top = 10
        self.width = 1380
        self.height = 700+200
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        ## Create master layout
        hbox = QHBoxLayout()
        
        ## Create plots
        canvas = PlotCanvas(self, width=12, height=7)
        #canvas.move(0,0)
        canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas = canvas
        
        canvas2 = PlotCanvas(self, width=12, height=2)
        #canvas2.move(0,700)
        self.canvas2 = canvas2
        
        # Create toolbar
        ## TODO there are some problems linking this to the actual canvas...
        toolbar = NavigationToolbar(self.canvas, self, False)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)
        self.toolbar = toolbar
        
        # Add to layout
        vbox = QVBoxLayout()
        vbox.addWidget(canvas)
        vbox.addWidget(canvas2)
        hbox.addLayout(vbox)

        ## Create parameter editing
        vbox = QVBoxLayout()
        
        # Save output
        button = QPushButton('Save Stitched Spectrum', self)
        button.setToolTip('')
        button.resize(120,50)
        button.clicked.connect(self.save_stitched_spectrum)
        self.button = button
        
        # Jump to order
        # TODO add a validator
        ledit_change_order = QLineEdit()
        btn_change_order = QPushButton('Change Order To', self)
        btn_change_order.setToolTip('')
        btn_change_order.resize(120,50)
        self.ledit_change_order = ledit_change_order
        self.btn_change_order = btn_change_order
        self.btn_change_order.clicked.connect(self.set_order_gui)
        self.ledit_change_order.returnPressed.connect(self.set_order_gui)
        
        # Change sigma lo/hi
        label_label = QLabel("Hello World",self)
        self.label_label = label_label
        label_siglo = QLabel("σlo",self)
        ledit_siglo = QLineEdit()
        label_sighi = QLabel("σhi",self)
        ledit_sighi = QLineEdit()
        self.ledit_siglo = ledit_siglo
        self.ledit_sighi = ledit_sighi
        self.ledit_siglo.returnPressed.connect(self.set_sigma_lo)
        self.ledit_sighi.returnPressed.connect(self.set_sigma_hi)
        self.btn_apply_sig_order = QPushButton("Apply To Order")
        self.btn_apply_sig_order.clicked.connect(self.apply_sigma_to_order)
        
        # Add to layout
        vbox.addWidget(button)
        
        vb = QVBoxLayout()
        vb.addWidget(btn_change_order)
        vb.addWidget(ledit_change_order)
        vbox.addLayout(vb)

        vb = QVBoxLayout()
        vb.addWidget(label_label)
        hb = QHBoxLayout()
        hb.addWidget(label_siglo); hb.addWidget(ledit_siglo)
        vb.addLayout(hb)
        hb = QHBoxLayout()
        hb.addWidget(label_sighi); hb.addWidget(ledit_sighi)
        vb.addLayout(hb)
        vb.addWidget(self.btn_apply_sig_order)
        vbox.addLayout(vb)
        
        hbox.addLayout(vbox)
        
        ## Add all this stuff to a QWidget wrapper
        self.window = QWidget(self)
        self.window.setLayout(hbox)
        self.setCentralWidget(self.window)
        
        self.activate_keyboard_shortcuts()

        self.canvas.setFocus()
        self.toolbar.update()
        self.show()
    
    def activate_keyboard_shortcuts(self):
        self._figure_key_press_cid = self.canvas.mpl_connect("key_press_event", self.figure_key_press)
        self._interactive_mask_region_signal = None
        
    def save_stitched_spectrum(self):
        fullspec = self.model.get_full_stitch()
        fullspec.write(self.stitched_default_fname)
        print("Wrote stitched spectrum to {}".format(self.stitched_default_fname))
    
    def figure_key_press(self, event):
        """
        Keyboard Shortcuts
        """
        #print("Pressed '{}'".format(event.key))
        ## Change orders (it is reversed on purpose!)
        if event.key in ("left", "j", "J", "right", "k", "K"):
            offset = -1 if event.key in ("right", "k", "K") else +1
            if (self.order + offset) in self.all_order_numbers:
                self.canvas.clear_plot()
                self.canvas2.clear_plot()
                self.set_order(self.order+offset)
            else:
                print("Order {} not available".format(self.order + offset))
            return
        elif event.key == "up":
            self.set_next_label()
        elif event.key == "down":
            self.set_prev_label()
        elif event.key == "1":
            self.decrement_sigma_lo()
        elif event.key == "2":
            self.increment_sigma_lo()
        elif event.key == "3":
            self.decrement_sigma_hi()
        elif event.key == "4":
            self.increment_sigma_hi()
        elif event.key == "0":
            self.apply_sigma_to_order()
        elif event.key == " ":
            # print mouse and knot location
            iknot, xknot, yknots = self.model.get_nearest_knot(self.order, event.xdata, gety=True)
            if xknot is not None:
                self.canvas.textinfo("x={:.1f} y={:.2f} kx={:.1f}".format(
                        event.xdata, event.ydata, xknot))
            else:
                self.canvas.textinfo("x={:.1f} y={:.2f} (no knots)".format(
                        event.xdata, event.ydata))
            xknots = [xknot for label in self.model.labels]
            colors = [self.colordict[label] for label in self.model.labels]
            #self.canvas.select_points(xknots, yknots, colors=colors)
            return
        elif event.key in "aA":
            # add knot
            self.model.add_knot(self.order, event.xdata)
            self.update_plot(reset_limits=False)
            return
        if event.key in "dD":
            # delete knot
            self.model.delete_nearest_knot(self.order, event.xdata)
            self.update_plot(reset_limits=False)
            return
        elif event.key in "cC":
            # refit continuum with default settings
            self.model.reset_exclude_regions(self.order)
            self.model.reset_knots(self.order)
            self.update_plot()
            return
        elif event.key in "rR":
            # redraw
            self.update_plot()
            return
        elif event.key in "fF":
            # fit
            self.fit_continuums()
            self.update_plot()
            return
        elif event.key in "eE":
            # exclude region
            xmin, xmax = self.canvas.ax.get_xlim()
            if xmin < event.xdata and event.xdata < xmax:
                self.start_interactive_exclude_region(event.xdata)
            return
        elif event.key in "sS":
            # save
            self.model.save(self.default_save)
            print("Saved to {}".format(self.default_save))
            return
        elif event.key in "qQ":
            # save and quit
            self.model.save(self.default_save)
            print("Saved to {}".format(self.default_save))
            self.close()
        elif event.key in "!":
            # quit without saving
            print("Quitting without saving (NOT IMPLEMENTED FOR SAFETY)")
            #self.close()
        elif event.key in "hH":
            print("Keyboard shortcuts:")
            print("left/right (j/k): increase/decrease order")
            print("up/down: increase/decrease label (for sigma)")
            print("1/2: decrease/increase siglo")
            print("3/4: decrease/increase sighi")
            print("0: apply siglo/sighi to all in this order")
            print("spacebar: print mouse and knot location")
            print("a: add knot")
            print("d: delete knot")
            print("r: redraw plot")
            print("f: refit")
            print("e: start exclude region")
            print("s: save to {}".format(self.default_save))
            print("q: save and quit")
            print("!: quit without saving (NOT IMPLEMENTED FOR SAFETY)")
    def start_interactive_exclude_region(self, x):
        # Disconnect all other signals
        self.canvas.mpl_disconnect(self._figure_key_press_cid)
        
        # Setup interactive mask in canvas
        xmin, xmax, ymin, ymax = (x, np.nan, -1e8, +1e8)
        for patch in self.canvas._interactive_mask:
            patch.set_xy([
                [xmin, ymin],
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
                [xmin, ymin]
            ])
            patch.set_facecolor("r")
        
        self._interactive_mask_region_signal = (
            time.time(),
            self.canvas.mpl_connect(
                "motion_notify_event", self.continue_interactive_exclude_region)
        )
        self._finish_interactive_exclude_region_cid = \
            self.canvas.mpl_connect("key_press_event", self.finish_interactive_exclude_region)
    def continue_interactive_exclude_region(self, event):
        if event.xdata is None: return None
        signal_time, signal_cid = self._interactive_mask_region_signal
        
        data = self.canvas._interactive_mask[0].get_xy()
        # Update xmax.
        data[2:4, 0] = event.xdata
        for patch in self.canvas._interactive_mask:
            patch.set_xy(data)
        self.canvas.draw()
        return None
    
    def finish_interactive_exclude_region(self, event):
        if event.key not in "eE": return
        
        signal_time, signal_cid = self._interactive_mask_region_signal
        
        xy = self.canvas._interactive_mask[0].get_xy()
        if event.xdata is None:
            # Out of axis; exclude based on the closest axis limit
            xdata = xy[2, 0]
        else:
            xdata = event.xdata
        
        # Save mask data
        minx = min(xy[0,0],xy[2,0])
        maxx = max(xy[0,0],xy[2,0])
        self.model.add_exclude_region(self.order, minx, maxx)

        # Clean up interactive mask
        xy[:, 0] = np.nan
        for patch in self.canvas._interactive_mask:
            patch.set_xy(xy)
        del self._interactive_mask_region_signal
        self.canvas.mpl_disconnect(signal_cid)
        self.canvas.mpl_disconnect(self._finish_interactive_exclude_region_cid)
        self._figure_key_press_cid = self.canvas.mpl_connect("key_press_event", self.figure_key_press)
        
        self.fit_continuums()
        self.update_plot(reset_limits=False)
        

class PlotCanvas(FigureCanvas):
    ## TODO put keyboard stuff in the canvas rather than the application?
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create canvas
        fig = Figure(figsize=(width, height), dpi=dpi)
        ax = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        self.setParent(parent)
        self.fig = fig
        self.ax = ax
        self._text = ax.text(.1, .1, "", ha='left', va='center',
                              transform=ax.transAxes)
        self._selected_points = ax.scatter([], [], edgecolor="b", facecolor="none", 
                                           s=150, linewidth=3, zorder=2)
        self._interactive_mask =[self.ax.axvspan(xmin=np.nan, xmax=np.nan, ymin=np.nan,
                                                ymax=np.nan, facecolor="r", edgecolor="none", alpha=0.25,
                                                zorder=-5),]
        self._vspans = []
        self._next_vspan = 0
        
    def textinfo(self, text):
        self._text.set_text(text)
        self.draw()
        
    def select_points(self, x, y, colors):
        self._selected_points.set_offsets(np.array([x, y]).T)
        self._selected_points.set_edgecolors(colors)
        #self.draw()
    def deselect_points(self):
        self._selected_points.set_offsets(np.array([[np.nan],[np.nan]]).T)
        #self.draw()

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)
        #self.draw()
    def axhline(self, *args, **kwargs):
        self.ax.axhline(*args, **kwargs)
        #self.draw()
    
    def vspan(self, xmin, xmax):
        try:
            patch = self._vspans[self._next_vspan]
        except IndexError:
            patch = self.ax.axvspan(xmin=xmin, xmax=xmax, facecolor="r", edgecolor="none", alpha=0.25, zorder=-5)
            self._vspans.append(patch)
        else:
            patch.set_xy([
                    [xmin, -1e8],
                    [xmin, +1e8],
                    [xmax, +1e8],
                    [xmax, -1e8],
                    [xmin, -1e8]
                    ])
            patch.set_visible(True)
        self._next_vspan += 1
        self.draw()
    
    def clear_plot(self):
        # Fix from Erika for too many lines
        for l in range(len(self.ax.lines)): self.ax.lines.pop(0)
        try:
            self.ax.legend_.remove()
        except:
            pass
        self.deselect_points()
        self._next_vspan = 0
        for patch in self._vspans:
            patch.set_xy([
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan]
                    ])
            patch.set_visible(False)
        self.draw()
    
    def reset_limits(self, plo=0, phi=100):
        self.set_xlim(get_ax_xlim(self.ax))
        self.set_ylim(get_ax_ylim(self.ax, plo=plo, phi=phi))
    def set_title(self, *args, **kwargs):
        self.ax.set_title(*args, **kwargs)
        self.draw()
    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)
        self.draw()
    def set_ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)
        self.draw()
    def get_xlim(self):
        return self.ax.get_xlim()
    def get_ylim(self):
        return self.ax.get_ylim()
    def legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)
        self.draw()
    
def get_ax_xlim(ax, plo=0, phi=100):
    allx = []
    for l in ax.lines:
        x = l.get_data()[0]
        allx.append(x)
    allx = np.concatenate(allx)
    xmin, xmax = np.nanpercentile(allx, [plo, phi])
    xmin = max(xmin, min(allx[np.isfinite(allx)]))
    xmax = min(xmax, max(allx[np.isfinite(allx)]))
    return xmin, xmax
def get_ax_ylim(ax, plo=0, phi=100):
    ally = []
    for l in ax.lines:
        y = l.get_data()[1]
        ally.append(y)
    ally = np.concatenate(ally)
    ymin, ymax = np.nanpercentile(ally, [plo, phi])
    ymin = max(ymin, min(ally[np.isfinite(ally)]))
    ymax = min(ymax, max(ally[np.isfinite(ally)]))
    return ymin, ymax

if __name__=="__main__":
    app = QApplication(sys.argv)
    basedir = "/Users/alexji/Dropbox/J0023+0307/order_coadd"
    prefixes = ["ani-noCR","ranaN5","tthN12","ian"]
    labels_red = ["{}_red".format(prefix) for prefix in prefixes]
    labels_blue = ["{}_blue".format(prefix) for prefix in prefixes]
    fnames_red = [os.path.join(basedir, "{}_j0023+0307red_multi.fits".format(prefix)) for prefix in prefixes]
    fnames_blue = [os.path.join(basedir, "{}_j0023+0307blue_multi.fits".format(prefix)) for prefix in prefixes]
    
    labels_red[0] = "ani-prune"
    fnames_red[0] = os.path.join(basedir, "{}_j0023+0307red_multi.fits".format("ani-prune"))
    labels = labels_red + labels_blue
    fnames = fnames_red + fnames_blue
    rvdict = {"ranaN5":-225.75, "ian":-223.50, "tthN12":-219.87, "ani-noCR":-207.81, "ani-prune":-207.81}
    rvcorr = [-rvdict[label.split("_")[0]] for label in labels]
    label_to_rv = dict(zip(labels, rvcorr))
    
    model = ContinuumModel(knot_spacing=15, sigma_lo=1.0, sigma_hi=0.5)
    model.load_data(fnames, labels=labels, fluxband=1, label_to_rv=label_to_rv)

    #model = ContinuumModel.load("sky_continuum_fit.npy")
    
    ex = ContinuumNormalizationApp(model)
    sys.exit(app.exec_())
    

def tmp():
    app = QApplication(sys.argv)
    model = ContinuumModel.load("J0023_continuum_fit.npy")
    ex = ContinuumNormalizationApp(model)
    sys.exit(app.exec_())
    #fig = plot_all_normalized_orders(model, 10, 7, marker=',', ls='none')
    #fig.savefig("test_plot_norm.png")
    
def tmp():
    #xlims = [(3855,3865),(3960,3980),(4290,4330),(5160,5190),(5885,5905),(6550,6570),(6700,6716)]
    #ylims = [(0.0, 1.2), (0.0, 1.2), (0.8, 1.1), (0.0, 1.2), (0.0, 1.2), (0.0, 1.2), (0.5, 1.1)]
    #fig = plot_normalized_order_regions(model, xlims, ylims, lw=.3, marker=',')
    #fig.savefig("test_plot_norm_region.png")
    xlims = [(6705,6711),(6700,6716)]
    ylims = [(0.8, 1.1),(0.5, 1.1)]
    fig = plot_normalized_order_regions(model, xlims, ylims, lw=.3, marker='.')
    fig.axes[1].legend()
    fig.savefig("Li_region.png")
    
    plt.show()
