#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An object for dealing with one-dimensional spectra. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from six import string_types

## Standard imports
import logging
import numpy as np
from scipy import interpolate, signal, ndimage
import os, sys, time
from collections import OrderedDict
import random
import functools

## Astro imports
from astropy.io import fits, ascii
from alexmods.specutils.spectrum import (Spectrum1D, read_mike_spectrum, stitch as spectrum_stitch)

## GUI imports
# https://pythonspot.com/pyqt5-matplotlib/
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QHBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QDialog, QLabel, QLineEdit, QWidget
from PyQt5.QtGui import QIcon, QIntValidator, QDoubleValidator
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mpl import MPLWidget

DOUBLE_CLICK_INTERVAL = 0.1 # MAGIC HACK

class SkySubModel(object):
    def __init__(self, wave, objfluxarr, skyfluxarr, initial_params=None, objnames=None):
        self.param_names = ["Scale","fwhm_obj","fwhm_sky","delta_lambda"]
        Npix = len(wave)
        Nobj, Nframe, _ = objfluxarr.shape
        assert _ == Npix
        assert objfluxarr.shape == skyfluxarr.shape
        if initial_params is None:
            initial_params = self.get_default_initial_params(Nobj, Nframe)
        if objnames is None:
            objnames = ["Obj{:03}".format(iobj) for iobj in range(Nobjr)]
        assert initial_params.shape == (Nobj, Nframe, 4)
        self.Nobj = Nobj
        self.Nframe = Nframe
        self.Nparam = 4

        objfluxarr[np.isnan(objfluxarr)] = 0.
        skyfluxarr[np.isnan(skyfluxarr)] = 0.
        self.wave = wave
        self.objfluxarr = objfluxarr
        self.skyfluxarr = skyfluxarr
        self.params = initial_params
        self.objnames = objnames
        
        # Assume linear wavelength spacing to within 0.01 pixels
        dwave = np.median(np.diff(wave))
        assert np.all(np.abs(np.diff(wave) - dwave) < dwave*0.01), (dwave, np.max(np.abs(np.diff(wave) - dwave)))
        self.dwave = dwave
    
    def save(self, fname):
        assert fname.endswith(".npy")
        np.save(fname, [self.wave, self.objfluxarr, self.skyfluxarr, self.params, self.objnames])
    @classmethod
    def load(cls, fname):
        assert fname.endswith(".npy")
        wave, objfluxarr, skyfluxarr, params, objnames = np.load(fname)
        return cls(wave, objfluxarr, skyfluxarr, params, objnames)
    
    def get_default_initial_params(self, Nobj, Nframe):
        params = np.zeros((Nobj, Nframe, 4))
        params[:,:,0] = 1.0
        return params
    
    def get_obj_flux(self, iobj, iframe):
        """ Apply FWHM and shift but not scale """
        flux = self.objfluxarr[iobj,iframe]
        fwhm = self.params[iobj, iframe, 1]
        shift = self.params[iobj, iframe, 3]
        return self.apply_params(flux, 1.0, fwhm, shift)
    def get_sky_flux(self, iobj, iframe):
        """ Apply FWHM and scale but not shift """
        flux = self.skyfluxarr[iobj,iframe]
        scale = self.params[iobj, iframe, 0]
        fwhm = self.params[iobj, iframe, 2]
        return self.apply_params(flux, scale, fwhm, 0.)
    def get_residual_flux(self, iobj, iframe):
        objflux = self.get_obj_flux(iobj, iframe)
        skyflux = self.get_sky_flux(iobj, iframe)
        return objflux - skyflux
    
    def apply_params(self, flux, scale, fwhm, shift):
        flux = self.apply_scaling(flux, scale)
        flux = self.apply_smoothing(flux, fwhm)
        flux = self.apply_shift(flux, shift)
        return flux
    def apply_scaling(self, flux, scale):
        return scale * flux
    def apply_smoothing(self, flux, fwhm):
        if fwhm==0: return flux
        sigma = fwhm / 2.355
        pixel_sigma = sigma / self.dwave
        return ndimage.gaussian_filter1d(flux, pixel_sigma)
    def apply_shift(self, flux, shift, fill_value=0.):
        new_wave = self.wave + shift
        new_flux = np.interp(new_wave, self.wave, flux, left=fill_value, right=fill_value)
        return new_flux

    def get_param_value(self,iobj,iframe,iparam):
        return self.params[iobj, iframe, iparam]
    def set_param_value(self,iobj,iframe,iparam, val):
        self.params[iobj, iframe, iparam] = val



class SkySubApp(QMainWindow):
    def __init__(self, modelfname, wavelength_ranges):
        super().__init__()

        self.wavelength_ranges = self.validate_wavelength_ranges(wavelength_ranges)
        self.num_figs = len(self.wavelength_ranges)
        self.initUI()
        self.new_model(modelfname)
        
    def validate_wavelength_ranges(self, wavelength_ranges):
        for x in wavelength_ranges:
            assert len(x) == 2
            assert x[1] > x[0]
        return wavelength_ranges

    def new_model(self, modelfname):
        model = SkySubModel.load(modelfname)
        self.model = model
        self.modelfname = modelfname
        self.canvas.set_wavelength_indices(model.wave)
        self.set_iobj_iframe(0, 0)
        
        self.edit_iobj.setValidator(QIntValidator(0, self.model.Nobj-1, self.edit_iobj))
        self.edit_iframe.setValidator(QIntValidator(0, self.model.Nframe-1, self.edit_iobj))

    def set_iobj_iframe(self, iobj, iframe):
        self.iobj = iobj
        self.iframe = iframe
        for ix in range(len(self.edit_list)):
            self.set_edit_value(ix,self.get_edit_value(ix))
        self.edit_iobj.setText(str(self.iobj))
        self.edit_iframe.setText(str(self.iframe))
        self.update_plot(relim_axes=True)
    
    def initUI(self):
        self.title = "Sky Subtraction"
        self.left = 10
        self.top = 10
        self.width = 1280
        self.height = 800

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        cw = QWidget(self)
        self.setCentralWidget(cw)
        vbox = QVBoxLayout(cw)
        vbox.setContentsMargins(10, 10, 10, 10)

        canvas = PlotCanvas(self.wavelength_ranges, self, width=11, height=8)
        canvas.move(0,0)
        canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas = canvas
        toolbar = NavigationToolbar(self.canvas, self, False)
        self.toolbar = toolbar
        vbox.addWidget(canvas)
        
        ## Start Button Row
        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        def _create_label_lineedit(text, vmin, vmax, dec=3, as_int=False):
            _vbox = QVBoxLayout()
            _vbox.setSpacing(0)
            #_vbox.setContentsMargins(0,0,0,0)
            label = QLabel(self)
            label.setText(text)
            line = QLineEdit(self)
            if as_int:
                line.setValidator(QIntValidator(vmin, vmax, line))
            else:
                line.setValidator(QDoubleValidator(vmin, vmax, dec, line))
            _vbox.addWidget(label)
            _vbox.addWidget(line)
            return _vbox, label, line
        # model parameters
        _vbox, label, line = _create_label_lineedit("Scale", -2, 2)
        self.edit_scale = line
        hbox.addLayout(_vbox)
        _vbox, label, line = _create_label_lineedit("FWHM Obj", 0, 10)
        self.edit_fwhm_obj = line
        hbox.addLayout(_vbox)
        _vbox, label, line = _create_label_lineedit("FWHM Sky", 0, 10)
        self.edit_fwhm_sky = line
        hbox.addLayout(_vbox)
        _vbox, label, line = _create_label_lineedit("Delta Wave", -5, 5)
        self.edit_offset = line
        hbox.addLayout(_vbox)
        # iobj, iframe
        _vbox, label, line = _create_label_lineedit("iobj", 0, 999)
        self.edit_iobj = line
        hbox.addLayout(_vbox)
        _vbox, label, line = _create_label_lineedit("iframe", 0, 999)
        self.edit_iframe = line
        hbox.addLayout(_vbox)
        ## End Button Row
        
        ## Interactivity
        self.edit_list = [self.edit_scale, self.edit_fwhm_obj, self.edit_fwhm_sky, self.edit_offset]
        self.set_edit_functions = [functools.partial(self.set_edit_value, edit_index=ix) for ix in range(4)]
        for line, func in zip(self.edit_list, self.set_edit_functions):
            line.textChanged.connect(func)
        self.edit_iobj.returnPressed.connect(self.edit_iobj_changed)
        self.edit_iframe.returnPressed.connect(self.edit_iframe_changed)
        self.activate_keyboard_shortcuts()
        
        self.canvas.setFocus()
        self.toolbar.update()
        self.show()
    
    def update_plot(self, relim_axes=False, draw=True):
        objflux = self.model.get_obj_flux(self.iobj, self.iframe)
        skyflux = self.model.get_sky_flux(self.iobj, self.iframe)
        resid = objflux - skyflux
        title = "{} Frame {}/{}".format(self.model.objnames[self.iobj],
                                            self.iframe+1, self.model.Nframe)
        self.canvas.plot(self.model.wave, objflux, skyflux, resid, relim_axes=relim_axes, draw=draw, title=title)

    def activate_keyboard_shortcuts(self):
        self._figure_key_press_cid = self.canvas.mpl_connect("key_press_event", self.figure_key_press)
        
    def edit_iobj_changed(self):
        new_iobj = int(self.edit_iobj.text())
        assert 0 <= new_iobj and new_iobj < self.model.Nobj
        self.iobj = new_iobj
        self.set_iobj_iframe(self.iobj, self.iframe)
        return None
    def edit_iframe_changed(self):
        new_iframe = int(self.edit_iframe.text())
        assert 0 <= new_iframe and new_iframe < self.model.Nframe
        self.iframe = new_iframe
        self.set_iobj_iframe(self.iobj, self.iframe)
        return None
    def get_edit_value(self, edit_index):
        """ Get LineEdit value from model """
        return self.model.get_param_value(self.iobj, self.iframe, edit_index)
    def set_edit_value(self, edit_index, val=None):
        """ Set LineEdit value in GUI and in model """
        if val is None:
            val = self.read_edit_value(edit_index)
        else:
            self.edit_list[edit_index].setText("{:.3f}".format(val))
        self.model.set_param_value(self.iobj, self.iframe, edit_index, val)
        self.update_plot()
        return None
    def read_edit_value(self, edit_index):
        try:
            val = float(self.edit_list[edit_index].text())
        except:
            return np.nan
        return val
    def increment_edit_value(self, edit_index, increment, minval=-np.inf, maxval=np.inf):
        """ Increment LineEdit value """
        val = self.get_edit_value(edit_index)
        val += increment
        self.set_edit_value(edit_index, min(max(val,minval),maxval))
        return None
    
    def figure_key_press(self, event):
        """
        Keyboard Shortcuts
        up/down: change sky scale
        left/right: change delta wave
        """
        if event.key == "left":
            self.increment_edit_value(3,+0.001)
            return
        if event.key == "right":
            self.increment_edit_value(3,-0.001)
            return
        if event.key == "up":
            self.increment_edit_value(0,+0.005)
            return
        if event.key == "down":
            self.increment_edit_value(0,-0.005)
            return
        if event.key in ("1"):
            self.increment_edit_value(1,-0.005,minval=0)
            return
        if event.key in ("2"):
            self.increment_edit_value(1,+0.005,minval=0)
            return
        if event.key in ("3"):
            self.increment_edit_value(2,-0.005,minval=0)
            return
        if event.key in ("4"):
            self.increment_edit_value(2,+0.005,minval=0)
            return
        if event.key in ("j","J"):
            new_iframe = (int(self.edit_iframe.text()) - 1) % self.model.Nframe
            self.set_iobj_iframe(self.iobj, new_iframe)
            return
        if event.key in ("k","K"):
            new_iframe = (int(self.edit_iframe.text()) + 1) % self.model.Nframe
            self.set_iobj_iframe(self.iobj, new_iframe)
            return
        if event.key in ("a","A"):
            new_iobj = (int(self.edit_iobj.text()) - 1) % self.model.Nobj
            self.set_iobj_iframe(new_iobj, self.iframe)
            return
        if event.key in ("e","E"):
            new_iobj = (int(self.edit_iobj.text()) + 1) % self.model.Nobj
            self.set_iobj_iframe(new_iobj, self.iframe)
            return
        elif event.key in ("r","R"):
            # redraw
            self.update_plot(relim_axes=True)
            return
        if event.key in ("s","S"):
            # save
            self.model.save(self.modelfname)
            print("Saved to {}".format(self.modelfname))
            return
        if event.key in ("q","Q"):
            # save and quit
            self.model.save(self.modelfname)
            print("Saved to {}".format(self.modelfname))
            self.close()
        if event.key in ("!"):
            # quit without saving
            print("Quitting without saving (NOT IMPLEMENTED FOR SAFETY)")
            #self.close()
            return

class PlotCanvas(FigureCanvas):
    ## TODO put keyboard stuff in the canvas rather than the application?
    def __init__(self, wavelength_ranges, parent=None, width=5, height=4, dpi=100):
        self.num_figs = len(wavelength_ranges)
        self.wavelength_ranges = wavelength_ranges
        
        # Create canvas
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        gs = gridspec.GridSpec(5, self.num_figs)
        gs.update(left=0.05, right=0.95,
                  bottom=0.05, top=0.95,
                  wspace=0.2,hspace=0)
        
        self.axes_flux = []
        self.axes_resid = []
        self.lines_objflux = []
        self.lines_skyflux = []
        self.lines_resid = []
        for i in range(self.num_figs):
            ax1 = fig.add_subplot(gs[0:3, i])
            ax2 = fig.add_subplot(gs[3:5, i], sharex=ax1)
            ax1.set_xlim(wavelength_ranges[i])
            ax1.set_xticklabels([])
            ax2.set_xlim(wavelength_ranges[i])
            self.axes_flux.append(ax1)
            self.axes_resid.append(ax2)
            l1, = ax1.plot([np.nan],[np.nan], c='k')
            l2, = ax1.plot([np.nan],[np.nan], c='c')
            l3, = ax2.plot([np.nan],[np.nan], c='k')
            self.lines_objflux.append(l1)
            self.lines_skyflux.append(l2)
            self.lines_resid.append(l3)
            ax2.axhline(0, color='k', ls=':', lw=1)
        self.setParent(parent)
        self.fig = fig
        self.title_text = fig.suptitle("")
        
        self.wave_buffer = 5.0
        self.wavelength_indices = None
    def set_wavelength_indices(self, wave):
        self.wavelength_indices = [(wave > w1-self.wave_buffer) & (wave < w2+self.wave_buffer) \
                                   for (w1, w2) in self.wavelength_ranges]
    def reset_wavelength_indices(self):
        self.wavelength_indices = None
    def plot(self, wave, objflux, skyflux, resid, relim_axes=False, draw=True, title=""):
        """ Plot in the wavelength ranges """
        if self.wavelength_indices is None:
            wavelength_indices = []
            for (w1,w2) in self.wavelength_ranges:
                iiplot = (wave > w1-self.wave_buffer) & (wave < w2+self.wave_buffer)
                wavelength_indices.append(iiplot)
        else:
            wavelength_indices = self.wavelength_indices
        for indices, ax1, ax2, l1, l2, l3 in zip(wavelength_indices, self.axes_flux, self.axes_resid,
                                                 self.lines_objflux, self.lines_skyflux, self.lines_resid):
            l1.set_data(wave[indices], objflux[indices])
            l2.set_data(wave[indices], skyflux[indices])
            l3.set_data(wave[indices], resid[indices])
            if relim_axes:
                ylim = get_ax_ylim(ax1)
                ax1.set_ylim(ylim)
                ylim = get_ax_ylim(ax2)
                ax2.set_ylim(ylim)
        self.title_text.set_text(title)
        if draw: self.draw()

    def clear_plot(self):
        for l1, l2, l3 in zip(self.lines_objflux, self.lines_skyflux, self.lines_resid):
            l1.set_data([np.nan], [np.nan])
            l2.set_data([np.nan], [np.nan])
            l3.set_data([np.nan], [np.nan])
        self.draw()
    
    #def reset_limits(self, plo=0, phi=100):
    #    self.set_xlim(get_ax_xlim(self.ax))
    #    self.set_ylim(get_ax_ylim(self.ax, plo=plo, phi=phi))
    #def set_title(self, *args, **kwargs):
    #    self.ax.set_title(*args, **kwargs)
    #    self.draw()
    #def set_xlim(self, *args, **kwargs):
    #    self.ax.set_xlim(*args, **kwargs)
    #    self.draw()
    #def set_ylim(self, *args, **kwargs):
    #    self.ax.set_ylim(*args, **kwargs)
    #    self.draw()
    #def legend(self, *args, **kwargs):
    #    self.ax.legend(*args, **kwargs)
    #    self.draw()
    
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
    finite = np.isfinite(ally)
    if finite.sum() == 0: return -1, 1
    ymin = max(ymin, np.min(ally[finite]))
    ymax = min(ymax, np.max(ally[finite]))
    return ymin, ymax

if __name__=="__main__":
    app = QApplication(sys.argv)
    modelfname = "/Users/alexji/Dropbox/RetIIMOS/M2FSdata/M2FS/b_skysubmodel_3.npy"
    wavelength_ranges = [(6460, 6480), (6495, 6500), (6530, 6540)]
    ex = SkySubApp(modelfname, wavelength_ranges)
    sys.exit(app.exec_())
