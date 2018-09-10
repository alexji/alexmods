#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An object for dealing with one-dimensional spectra. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from six import string_types

__all__ = ["fit_continuum"]

## Standard imports
import logging
import numpy as np
from scipy import interpolate, signal
import os, sys
from collections import OrderedDict
import random

## Astro imports
from astropy.io import fits, ascii
from alexmods.specutils.spectrum import Spectrum1D, read_mike_spectrum

## GUI imports
# https://pythonspot.com/pyqt5-matplotlib/
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def median_filter(spec, window):
    newflux = signal.median_filter(spec.flux, window)
    return Spectrum1D(spec.dispersion, newflux, spec.ivar, spec.metadata)

def fit_continuum_lsq(spec, knots, exclude=[], **kwargs):
    """ Fit least squares continuum through spectrum data using specified knots, return model """
    assert np.all(np.array(list(map(len, exclude))) == 2), exclude
    assert np.all(np.array(list(map(lambda x: x[0] < x[1], exclude)))), exclude
    x, y, w = spec.dispersion, spec.flux, spec.ivar
    mask = np.ones_like(x, dtype=bool)
    for xmin, xmax in exclude:
        mask[(x >= xmin) & (x <= xmax)] = False
    try:
        fcont = interpolate.LSQUnivariateSpline(x[mask], y[mask], knots, w=w[mask], **kwargs)
    except ValueError:
        print("Knots:",knots)
        print("xmin, xmax = {:.4f}, {:.4f}".format(x[mask].min(), x[mask].max()))
        raise
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
    waverange = wmax - wmin
    Nknots = int(waverange // knot_spacing)
    minknot = wmin + (waverange - Nknots * knot_spacing)/2.
    xknots = np.arange(minknot, wmax, knot_spacing)
    # Make sure there the knots don't hit the edges
    while xknots[-1] >= wmax: xknots = xknots[:-1]
    while xknots[0] <= wmin: xknots = xknots[1:]
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

def fit_continuum_to_spectra(specs, knots, **kwargs):
    """ Run lsq spline fit on all spectra """
    cont_funcs = OrderedDict()
    cont_data = OrderedDict()
    for label, spec in specs.items():
        fcont = fit_continuum_lsq(spec, knots, **kwargs)
        cont_funcs[label] = fcont
        dcont = fcont(spec.dispersion)
        cont_data[label] = dcont
    return cont_funcs, cont_data

class ContinuumModel(object):
    def __init__(self, degree=3, knot_spacing=10.):
        self.degree = degree
        self.knot_spacing = knot_spacing
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
        
    def load_data(self, fnames, labels=None, fluxband=7):
        """
        Load data, sort by order number
        """
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
                    self.all_y_knots[iord] = []
                    
                    self.all_knots[iord] = []
                    self.all_exclude_regions[iord] = []
                self.all_specs[iord][label] = spec
                self.all_continuum_data[iord][label] = None
                self.all_continuum_functions[iord][label] = None
        self.all_order_numbers = list(np.sort(all_specs.keys()))
        
        self.fnames = fnames
        self.labels = labels
        return
    
    def fit_all_continuums(self):
        """ Fit all data for all orders """
        for order in self.all_order_numbers:
            self.fit_continuums(order)
        return
    def fit_continuums(self, order):
        """ Fit all data for one order """
        specs = self.all_specs[order]
        knots = self.get_knots(order)
        exclude = self.exclude_regions[order]
        fconts, dconts = fit_continuum_to_spectra(specs, knots, k=self.degree, exclude=exclude)
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
    def get_nearest_knot(self, order, x):
        knots = self.all_knots[order]
        iknot = np.argmin(np.abs(x-np.array(knots)))
        return iknot, knots[iknot]
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
    
class ContinuumNormalizationApp(QMainWindow):
    def __init__(self, input_spectra_filenames, labels=None, fluxband=7, **kwargs):
        super().__init__()

        self.model = ContinuumModel(**kwargs)
        self.model.load_data(input_spectra_filenames, labels=labels, fluxband=fluxband)
        self.labels = labels
        
        self.all_orders = self.model.all_specs
        self.all_order_numbers = self.all_order_numbers = self.model.all_order_numbers
        
        # Setup label colors
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.colordict = {}
        for i,label in enumerate(model.labels):
            self.colordict[label] = colors[i]
        
        self.all_knots = self.model.all_knots
        self.all_y_knots = self.model.all_y_knots
        self.all_continuum_data = self.model.all_continuum_data
        self.all_continuum_functions = self.model.all_continuum_functions
        self.all_exclude_regions = self.model.all_exclude_regions
        
        self.title = "Continuum Normalization"
        self.left = 10
        self.top = 10
        self.width = 1280
        self.height = 800
        self.initUI()
        
        self.set_order(66)
        
    def fit_all_continuums(self):
        self.model.fit_all_continuums()

    def set_order(self, order):
        """ Set which order to plot """
        assert order in self.all_order_numbers
        self.order = order
        self.canvas.set_title("Order {}".format(self.order))
        self.update_plot()
    
    def update_plot(self):
        self.canvas.clear_plot()
        self.plot_data()
        self.plot_continuums()
        self.plot_exclude_regions()
        self.canvas.draw()
    def plot_data(self):
        self.canvas.clear_plot()
        specs = self.all_orders[self.order]
        for label, spec in specs.items():
            self.canvas.plot(spec.dispersion, spec.flux, ',-', label=label, lw=.2,
                             color=self.colordict[label])
        self.canvas.reset_limits()
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
        except:
            print("Error plotting continuum {}".format(self.order))
    def plot_exclude_regions(self):
        pass
                
    
    def update_y_knots(self):
        if self.order in self.all_knots:
            specs = self.all_orders[self.order]
            conts = self.all_continuum_data[self.order]
            knots = self.all_knots[self.order]
            if self.order not in self.all_y_knots:
                self.all_y_knots[self.order] = OrderedDict()
            y_knots = self.all_y_knots[self.order]
            for label, spec in specs.items():
                cont = conts[label]
                ixknots = np.searchsorted(spec.dispersion, knots)
                y_knots[label] = cont[ixknots]
        
    def fit_continuums(self):
        """
        Fit continuum through knots to all spectra of this order
        """
        # Get spectra and knots
        specs = self.all_orders[self.order]
        if self.order in self.all_knots and len(self.all_knots[self.order]) > 0:
            knots = self.all_knots[self.order]
        else:
            logging.info("Initializing knots with spacing {}".format(self.knot_spacing))
            knots = initialize_knots_from_spectra(specs, self.knot_spacing)
            self.all_knots[self.order] = knots
        
        fconts, dconts = fit_continuum_to_spectra(specs, knots, k=self.degree)
        self.all_continuum_functions[self.order] = fconts
        self.all_continuum_data[self.order] = dconts
        self.update_y_knots()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        canvas = PlotCanvas(self, width=11, height=8)
        canvas.move(0,0)
        canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas = canvas
        
        # Create toolbar
        ## TODO there are some problems linking this to the actual canvas...
        toolbar = NavigationToolbar(self.canvas, self, False)
        self.toolbar = toolbar
        
        button = QPushButton('Fit Continuums', self)
        button.setToolTip('This is an example button')
        button.move(1100,0)
        button.resize(120,50)
        #button.clicked.connect(self.plot_data)
        button.clicked.connect(self.fit_continuums)
        button.clicked.connect(self.update_plot)
        self.button = button
        
        self.activate_keyboard_shortcuts()

        self.canvas.setFocus()
        self.toolbar.update()
        self.show()
    
    def activate_keyboard_shortcuts(self):
        self.canvas.mpl_connect("key_press_event", self.figure_key_press)
    def figure_key_press(self, event):
        print("Pressed '{}'".format(event.key))
        ## Change orders
        if event.key in ("left", "right"):
            offset = 1 if event.key == "right" else -1
            if (self.order + offset) in self.all_order_numbers:
                self.canvas.clear_plot()
                self.set_order(self.order+offset)
            return
        if event.key == " ":
            # print mouse and knot location
            iknot, xknot = self.get_nearest_knot(event.xdata)
            if xknot is not None:
                self.canvas.textinfo("x={:.1f} y={:.2f} kx={:.1f}".format(
                        event.xdata, event.ydata, xknot))
            else:
                self.canvas.textinfo("x={:.1f} y={:.2f} (no knots)".format(
                        event.xdata, event.ydata))
            # TODO select knot visually
            return
        if event.key in "aA":
            # add knot
            self.add_knot(event.xdata)
            self.update_plot()
            return
        if event.key in "dD":
            # delete knot
            iknot, xknot = self.get_nearest_knot(event.xdata)
            self.delete_knot(iknot)
            self.update_plot()
            return
        if event.key in "cC":
            # refit continuum with default settings
            if self.order in self.all_knots:
                # Remove all knots
                _ = self.all_knots.pop(self.order)
                self.fit_continuums()
                self.update_plot()
            return
        if event.key in "rR":
            # redraw
            if self.order in self.all_knots:
                self.update_plot()
            return
        if event.key in "eE":
            # exclude region
            xmin, xmax = self.canvas.ax.get_xlim()
            if xmin < event.xdata and event.data < xmax:
                self.start_interactive_exclude_region()
            return
    def start_interactive_exclude_region(self):
        self.canvas.mpl_disconnect("key_press_event", self.figure_key_press)
        self.canvas.mpl_connect("key_press_event", self.finish_interactive_exclude_region)
    def continue_interactive_exclude_region(self, event):
        pass
    def finish_interactive_exclude_region(self, event):
        xmin, xmax = self.canvas.ax.get_xlim()
        if xmin < event.xdata and event.xdata < xmax:
        self.canvas.mpl_disconnect("key_press_event", self.finish_interactive_exclude_region)
        self.canvas.mpl_connect("key_press_event", self.figure_key_press)
        pass
            
    def add_knot(self, x):
        if self.order not in self.all_knots: return
        knots = self.all_knots[self.order]
        ix = np.searchsorted(knots, x)
        knots = list(knots)
        knots.insert(ix, x)
        assert np.allclose(np.sort(knots), np.array(knots))
        self.all_knots[self.order] = np.array(knots)
        self.fit_continuums()
        return
    def delete_knot(self, iknot):
        if iknot is None: return
        knots = self.all_knots[self.order]
        knots = list(knots)
        knots.pop(iknot)
        assert np.allclose(np.sort(knots), np.array(knots))
        self.all_knots[self.order] = np.array(knots)
        self.fit_continuums()
        return
    def get_nearest_knot(self, x):
        """ Given coordinate x, find the nearest knot """
        if self.order not in self.all_knots:
            logging.debug("This order has no knots")
            return None, None
        # Get normalized location
        knots = self.all_knots[self.order]
        xmin, xmax = self.canvas.ax.get_xlim()
        normknots = (knots - xmin)/(xmax-xmin)
        normx = (x - xmin)/(xmax-xmin)
        iknot = np.argmin(np.abs(normx-normknots))
        return iknot, knots[iknot]
    
        
class PlotCanvas(FigureCanvas):
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
        self._interactive_mask =self.ax_spectrum.axvspan(xmin=np.nan, xmax=np.nan, ymin=np.nan,
                                                         ymax=np.nan, facecolor="r", edgecolor="none", alpha=0.25,
                                                         zorder=-5)
        
    def textinfo(self, text):
        self._text.set_text(text)
        self.draw()
        
    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)
        self.draw()
    def plot_norefresh(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)
    def clear_plot(self):
        for l in self.ax.lines:
            l.remove()
        try:
            self.ax.legend_.remove()
        except:
            pass
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    fname1 = "/Users/alexji/Dropbox/J0023+0307/order_coadd/ani-noCR_j0023+0307red_multi.fits"
    fname2 = "/Users/alexji/Dropbox/J0023+0307/order_coadd/ranaN5_j0023+0307red_multi.fits"
    fnames = [fname1, fname2]
    ex = ContinuumNormalizationApp(fnames)
    sys.exit(app.exec_())
    
