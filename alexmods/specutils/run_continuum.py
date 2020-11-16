"""
This is a script to run the continuum GUI.
Make a copy in your project working directory and edit as needed.
Be careful about running it within the package directory, it can mess up imports and stuff.

To use:
- edit this script to point to the files you want
- edit this script's label names to match the files
- measure the radial velocities by hand beforehand, e.g. with IRAF or SMHR and put them in this script
- python run_continuum.py

This continuum GUI is inspired by merging X Prochaska's linetools and SMHR by Andy Casey and Alex Ji.
It takes MIKE spectra, sorts them by order, and then lets you exclude regions and add/delete knots.

Keyboard shortcuts:
  - h: print help to the terminal
  - left/right or j/k: change the order to lower/higher wavelengths
  - a: add a knot at the x-location of the cursor
  - d: delete a knot closest to the x-location of the cursor
  - e: start/stop a mask
  - s: save
  - q: save and quit

  - up/down: cycle the currently selected spectrum
  - 1/2: decrease/increase sigma_lo by 0.1 for currently selected spectrum
  - 3/4: decrease/increase sigma_hi by 0.1 for currently selected spectrum
  - 0: apply current spectrum's sigma to every spectrum in the current order
  - c: clear all masks and knots and restart from default
  - r: redraw the plot
  - f: refit the continuum and redraw the plot (not often used)
"""

import numpy as np
import sys
from alexmods.specutils.continuum import ContinuumModel, ContinuumNormalizationApp
from PyQt5.QtWidgets import QApplication

if __name__=="__main__":
    app = QApplication(sys.argv)
    
    """
    This is an example of loading red and blue spectra from two different observing runs into one continuum model.
    I manually measured the observed velocities in another program, then specify them here.
    """
    ## Paths to spectra
    basedir = "/path/to/my/stars"
    fnames = ["star-expAblue_multi.fits","star-expBblue_multi.fits",
              "star-expAred_multi.fits","star-expBred_multi.fits"] 
    fnames = [basedir+"/"+x for x in fnames]
    ## This is just a label for each spectrum for displaying within the program
    labels = ["starA_b","starB_b",
              "starA_r","starB_r"]
    ## The label_to_rv dictionary goes from the spectrum's label to the velocity correction to apply to the spectra
    velocities = [+100.2, +103.7, +100.2, +103.7] # have to measure this separately
    velocities = [-v for v in velocities] # reverse the sign to get the correction to apply
    label_to_rv = dict(zip(labels,velocities))
    
    """
    This creates the continuum model and loads in the spectrum data.
    The parameters here are the defaults to use for each order.
    It's optimized for a low-S/N metal-poor star.
    I would normalize a few orders manually to find a good initialization for your star.
    """
    model = ContinuumModel(knot_spacing=20, sigma_lo=3.0, sigma_hi=0.3)
    model.load_data(fnames, labels=labels, fluxband=2, label_to_rv=label_to_rv)
    """
    If desired, you can load the continuum parameters from another star's saved model.
    This keeps the knot locations, masks, sigma clipping, etc.
    Very useful if normalizing a few stars of the same type.
    """
    #model.load_parameters_from("other_star_continuum_fit.npy")
    
    """
    *Alternatively* to creating a new ContinuumModel and loading in data,
    you can load a previously saved continuum model.
    """
    #model = ContinuumModel.load("continuum_fit.npy")
    
    """
    This runs the GUI
    """
    ex = ContinuumNormalizationApp(model)
    sys.exit(app.exec_())
