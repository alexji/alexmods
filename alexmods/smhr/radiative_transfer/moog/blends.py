#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Functionality to derive abundances from a transition's curve-of-growth using the
MOOG(SILENT) radiative transfer code in a blended transition
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import logging
import numpy as np
import re
import yaml
from pkg_resources import resource_stream

from . import utils
from .utils import RTError
from ....smhutils import element_to_species
from .... import linelists

logger = logging.getLogger(__name__)

# Load the MOOG defaults.
with resource_stream(__name__, "defaults.yaml") as fp:
    try:
        _moog_defaults = yaml.load(fp, yaml.FullLoader)
    except AttributeError:
        _moog_defaults = yaml.load(fp)

def blends_cog(photosphere, transitions, element, eqw, full_output=False, verbose=False,
               isotopes=None, twd=None, **kwargs):
    """
    Calculate atomic line abundances by interpolating the measured 
    equivalent width from the curve-of-growth. 
    This wraps the MOOG `blends` driver.

    :param photosphere:
        A formatted photosphere.

    :param transitions:
        A list of atomic transitions with measured equivalent widths.

    :param element:
        The element (Z) whose abundance to vary in order to match the line

    :param eqw:
        The equivalent width to fit

    :param verbose: [optional]
        Specify verbose flags to MOOG. This is primarily used for debugging.
    """

    # Create a temporary directory.
    path = utils.twd_path(twd=twd,**kwargs)

    # Write out the photosphere.
    moog_in, model_in, lines_in \
        = path("batch.par"), path("model.in"), path("lines.in")
    photosphere.write(model_in, format="moog")
    
    
    # Write out the transitions.
    #assert transitions[0]["wavelength"] > 0, transitions["wavelength"]
    #assert np.all(transitions[1:]["wavelength"] < 0), transitions["wavelength"]
    transitions["equivalent_width"] = eqw
    transitions[0]["equivalent_width"] = eqw
    # note that this must write out the EW too
    transitions.write(lines_in, format="moog")
    
    # Load the blends driver template.
    with resource_stream(__name__, "blends.in") as fp:
        template = fp.read().decode("utf-8")

    # Not these are SMH defaults, not MOOG defaults.
    kwds = _moog_defaults.copy()
    if verbose:
        kwds.update({
            "atmosphere": 2,
            "molecules": 2,
            "lines": 3, # 4 is max verbosity, but MOOG falls over.
        })

    # Parse keyword arguments.
    kwds.update(kwargs)

    # Parse I/O files:
    kwds.update({
        "element_to_fit":element,
        "standard_out": path("blends.std.out"),
        "summary_out": path("blends.sum.out"),
        "model_in": model_in,
        "lines_in": lines_in,
    })
    
    # Isotopes.
    kwds["isotopes_formatted"] = utils._format_isotopes(isotopes, 
        kwargs.pop("isotope_ionisation_states", (0, 1)), num_synth=1)
    
    contents = template.format(**kwds)

    # Write this to batch.par
    with open(moog_in, "w") as fp:
        fp.write(contents)

    # Execute MOOG in the TWD.
    code, out, err = utils.moogsilent(moog_in, **kwargs)
    out = out.decode("utf-8")
    err = err.decode("utf-8")

    if verbose:
        print("CODE:",code)
        print("OUT:\n", strip_control_characters(out))
        print("ERR:\n", strip_control_characters(err))

    # Returned normally?
    if code != 0:
        logger.error("MOOG returned the following standard output:")
        logger.error(out)
        logger.error("MOOG returned the following errors (code: {0:d}):".format(code))
        logger.error(err)
        logger.exception(RTError(err))
    else:
        logger.info("MOOG executed {0} successfully".format(moog_in))

    # Parse the output.
    abund = _parse_blends_summary(kwds["summary_out"])
    return abund

    ## TODO go from here: return 
    #raise NotImplementedError
    

def strip_control_characters(out):
    for x in np.unique(re.findall(r"\x1b\[K|\x1b\[\d+;1H",out)):
        out = out.replace(x,'')
    return out

def _parse_blends_summary(summary_out_path):
    """
    Parse output from the summary output find from the MOOGSILENT `blends`
    driver.

    :param summary_out_path:
        The path of the summary output file from the `blends` driver.
    """
    with open(summary_out_path, "r") as fp:
        summary = fp.readlines()
    abund = np.nan
    for line in summary:
        if not line.startswith("average abundance ="): continue
        s = line.split()
        abund = float(s[3])
        nlines = int(s[10])
        if nlines != 1:
            logger.warn("More than one line was measured!")
        break
    return abund
    #raise NotImplementedError
