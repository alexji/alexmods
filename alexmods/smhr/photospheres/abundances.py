#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Solar abundances """

from __future__ import division, absolute_import, print_function, unicode_literals


__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

from numpy import array
from .utils import element

from ...smhutils import element_to_atomic_number, species_to_element

_asplund_2009 = {
    "Pr": 0.72, 
    "Ni": 6.22, 
    "Gd": 1.07, 
    "Pd": 1.57, 
    "Pt": 1.62, 
    "Ru": 1.75, 
    "S": 7.12, 
    "Na": 6.24, 
    "Nb": 1.46, 
    "Nd": 1.42, 
    "Mg": 7.6, 
    "Li": 1.05, 
    "Pb": 1.75, 
    "Re": 0.26, 
    "Tl": 0.9, 
    "Tm": 0.1, 
    "Rb": 2.52, 
    "Ti": 4.95, 
    "As": 2.3, 
    "Te": 2.18, 
    "Rh": 0.91, 
    "Ta": -0.12, 
    "Be": 1.38, 
    "Xe": 2.24, 
    "Ba": 2.18, 
    "Tb": 0.3, 
    "H": 12.0, 
    "Yb": 0.84, 
    "Bi": 0.65, 
    "W": 0.85, 
    "Ar": 6.4, 
    "Fe": 7.5, 
    "Br": 2.54, 
    "Dy": 1.1, 
    "Hf": 0.85, 
    "Mo": 1.88, 
    "He": 10.93, 
    "Cl": 5.5, 
    "C": 8.43, 
    "B": 2.7, 
    "F": 4.56, 
    "I": 1.55, 
    "Sr": 2.87, 
    "K": 5.03, 
    "Mn": 5.43, 
    "O": 8.69, 
    "Ne": 7.93, 
    "P": 5.41, 
    "Si": 7.51, 
    "Th": 0.02, 
    "U": -0.54, 
    "Sn": 2.04, 
    "Sm": 0.96, 
    "V": 3.93, 
    "Y": 2.21, 
    "Sb": 1.01, 
    "N": 7.83, 
    "Os": 1.4, 
    "Se": 3.34, 
    "Sc": 3.15, 
    "Hg": 1.17, 
    "Zn": 4.56, 
    "La": 1.1, 
    "Ag": 0.94, 
    "Kr": 3.25, 
    "Co": 4.99, 
    "Ca": 6.34, 
    "Ir": 1.38, 
    "Eu": 0.52, 
    "Al": 6.45, 
    "Ce": 1.58, 
    "Cd": 1.71, 
    "Ho": 0.48, 
    "Ge": 3.65, 
    "Lu": 0.1, 
    "Au": 0.92, 
    "Zr": 2.58, 
    "Ga": 3.04, 
    "In": 0.8, 
    "Cs": 1.08, 
    "Cr": 5.64, 
    "Cu": 4.19, 
    "Er": 0.92,
    "Tc": -5.0 # MOOG uses this
}

_asplund_2020 = {"H":12.00,
                 "He":10.914,
                 "Li":0.96,
                 "Be":1.38,
                 "B":2.70,
                 "C":8.46,
                 "N":7.83,
                 "O":8.69,
                 "F":4.40,
                 "Ne":8.06,
                 "Na":6.22,
                 "Mg":7.55,
                 "Al":6.43,
                 "Si":7.51,
                 "P":5.41,
                 "S":7.12,
                 "Cl":5.31,
                 "Ar":6.38,
                 "K":5.07,
                 "Ca":6.30,
                 "Sc":3.14,
                 "Ti":4.97,
                 "V":3.90,
                 "Cr":5.62,
                 "Mn":5.42,
                 "Fe":7.46,
                 "Co":4.94,
                 "Ni":6.20,
                 "Cu":4.18,
                 "Zn":4.56,
                 "Ga":3.02,
                 "Ge":3.62,
                 "As":2.30,
                 "Se":3.34,
                 "Br":2.54,
                 "Kr":3.12,
                 "Rb":2.32,
                 "Sr":2.83,
                 "Y":2.21,
                 "Zr":2.59,
                 "Nb":1.47,
                 "Mo":1.88,
                 "Ru":1.75,
                 "Rh":0.78,
                 "Pd":1.57,
                 "Ag":0.96,
                 "Cd":1.71,
                 "In":0.80,
                 "Sn":2.02,
                 "Sb":1.01,
                 "Te":2.18,
                 "I":1.55,
                 "Xe":2.22,
                 "Cs":1.08,
                 "Ba":2.27,
                 "La":1.11,
                 "Ce":1.58,
                 "Pr":0.75,
                 "Nd":1.42,
                 "Sm":0.95,
                 "Eu":0.52,
                 "Gd":1.08,
                 "Tb":0.31,
                 "Dy":1.10,
                 "Ho":0.48,
                 "Er":0.93,
                 "Tm":0.11,
                 "Yb":0.85,
                 "Lu":0.10,
                 "Hf":0.85,
                 "Ta":-0.15,
                 "W":0.79,
                 "Re":0.26,
                 "Os":1.35,
                 "Ir":1.32,
                 "Pt":1.61,
                 "Au":0.91,
                 "Hg":1.17,
                 "Tl":0.92,
                 "Pb":1.95,
                 "Bi":0.65,
                 "Th":0.03,
                 "Tc": -5.0 # MOOG uses this
}
    
def asplund_2009(elements):
    """
    Return the Asplund 2009 solar abundance for the given element(s).

    :param elements:
        Input elements provided as integer-like or string representations.

    :type element:
        int, str, or list/array-like of integer-like objects

    :returns:
        The abundances of the input elements given the Asplund (2009) solar
        composition.
    """

    def parse(x):

        if isinstance(x, (str, bytes)):
            try:
                return (_asplund_2009[x], True)
            except KeyError:
                # It's a molecule, get the "good" Z
                Z = element_to_atomic_number(x)
                return (_asplund_2009[element(Z)], True)

        elif isinstance(x, (int, float)):
            try:
                el = element(x)
            except IndexError: # Molecules
                el = species_to_element(x).split()[0]

            try:
                return (_asplund_2009[el], True)
            except KeyError:
                # It's a molecule, get the "good" element name
                molecule = species_to_element(x)
                assert "-" in molecule, "Input {}, molecule {}".format(x, molecule)
                Z = element_to_atomic_number(molecule)
                return (_asplund_2009[element(Z)], True)

        else:
            # Assume list-type
            return ([parse(el)[0] for el in x], False)

    abundances, is_scalar = parse(elements)

    if is_scalar:
        return abundances
        
    return array(abundances)

def asplund_2020(elements):
    """
    Return the Asplund 2020 solar abundance for the given element(s).

    :param elements:
        Input elements provided as integer-like or string representations.

    :type element:
        int, str, or list/array-like of integer-like objects

    :returns:
        The abundances of the input elements given the Asplund (2020) solar
        composition.
    """

    def parse(x):

        if isinstance(x, (str, bytes)):
            try:
                return (_asplund_2020[x], True)
            except KeyError:
                # It's a molecule, get the "good" Z
                Z = element_to_atomic_number(x)
                return (_asplund_2020[element(Z)], True)

        elif isinstance(x, (int, float)):
            try:
                el = element(x)
            except IndexError: # Molecules
                el = species_to_element(x).split()[0]

            try:
                return (_asplund_2020[el], True)
            except KeyError:
                # It's a molecule, get the "good" element name
                molecule = species_to_element(x)
                assert "-" in molecule, "Input {}, molecule {}".format(x, molecule)
                Z = element_to_atomic_number(molecule)
                return (_asplund_2020[element(Z)], True)

        else:
            # Assume list-type
            return ([parse(el)[0] for el in x], False)

    abundances, is_scalar = parse(elements)

    if is_scalar:
        return abundances
        
    return array(abundances)
