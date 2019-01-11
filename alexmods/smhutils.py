# coding: utf-8

""" Utility functions from Spectroscopy Made Hard """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

# Standard library
import os
import logging
import platform
import string
import sys
import traceback
import tempfile

from collections import Counter

from six import string_types

from hashlib import sha1 as sha
from random import choice
from socket import gethostname, gethostbyname

# Third party imports
import numpy as np
import astropy.table

common_molecule_name2Z = {
    'Mg-H': 12,'H-Mg': 12,
    'C-C':  6,
    'C-N':  7, 'N-C':  7, #TODO
    'C-H':  6, 'H-C':  6,
    'O-H':  8, 'H-O':  8,
    'Fe-H': 26,'H-Fe': 26,
    'N-H':  7, 'H-N':  7,
    'Si-H': 14,'H-Si': 14,
    'Ti-O': 22,'O-Ti': 22,
    'V-O':  23,'O-V':  23,
    'Zr-O': 40,'O-Zr': 40
    }
common_molecule_name2species = {
    'Mg-H': 112,'H-Mg': 112,
    'C-C':  606,
    'C-N':  607,'N-C':  607,
    'C-H':  106,'H-C':  106,
    'O-H':  108,'H-O':  108,
    'Fe-H': 126,'H-Fe': 126,
    'N-H':  107,'H-N':  107,
    'Si-H': 114,'H-Si': 114,
    'Ti-O': 822,'O-Ti': 822,
    'V-O':  823,'O-V':  823,
    'Zr-O': 840,'O-Zr': 840
    }
common_molecule_species2elems = {
    112: ["Mg", "H"],
    606: ["C", "C"],
    607: ["C", "N"],
    106: ["C", "H"],
    108: ["O", "H"],
    126: ["Fe", "H"],
    107: ["N", "H"],
    114: ["Si", "H"],
    822: ["Ti", "O"],
    823: ["V", "O"],
    840: ["Zr", "O"]
    }

__all__ = ["element_to_species", "element_to_atomic_number", "species_to_element", "atomic_number_to_element", 
           "get_common_letters",
           "elems_isotopes_ion_to_species", "species_to_elems_isotopes_ion",
           "find_common_start", "extend_limits"]

logger = logging.getLogger(__name__)


def mkdtemp(**kwargs):
    if not os.path.exists(os.environ["HOME"]+"/.smh"):
        logger.info("Making "+os.environ["HOME"]+"/.smh")
        os.mkdir(os.environ["HOME"]+"/.smh")
    if 'dir' not in kwargs:
        kwargs['dir'] = os.environ["HOME"]+"/.smh"
    return tempfile.mkdtemp(**kwargs)
def mkstemp(**kwargs):
    if not os.path.exists(os.environ["HOME"]+"/.smh"):
        logger.info("Making "+os.environ["HOME"]+"/.smh")
        os.mkdir(os.environ["HOME"]+"/.smh")
    if 'dir' not in kwargs:
        kwargs['dir'] = os.environ["HOME"]+"/.smh"
    return tempfile.mkstemp(**kwargs)

def random_string(N=10):
    return ''.join(choice(string.ascii_uppercase + string.digits) for _ in range(N))

# List the periodic table here so that we can use it outside of a single
# function scope (e.g., 'element in utils.periodic_table')

periodic_table = """H                                                  He
                    Li Be                               B  C  N  O  F  Ne
                    Na Mg                               Al Si P  S  Cl Ar
                    K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                    Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                    Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                    Fr Ra Lr Rf"""

lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"

periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
    .replace(" Ra ", " Ra " + actinoids + " ").split()
del actinoids, lanthanoids



def element_to_species(element_repr):
    """ Converts a string representation of an element and its ionization state
    to a floating point """
    
    if not isinstance(element_repr, string_types):
        raise TypeError("element must be represented by a string-type")
        
    if element_repr.count(" ") > 0:
        element, ionization = element_repr.split()[:2]
    else:
        element, ionization = element_repr, "I"
    
    if element not in periodic_table:
        try:
            return common_molecule_name2species[element]
        except KeyError:
            # Don't know what this element is
            return float(element_repr)
    
    ionization = max([0, ionization.upper().count("I") - 1]) /10.
    transition = periodic_table.index(element) + 1 + ionization
    return transition


def element_to_atomic_number(element_repr):
    """
    Converts a string representation of an element and its ionization state
    to a floating point.

    :param element_repr:
        A string representation of the element. Typical examples might be 'Fe',
        'Ti I', 'si'.
    """
    
    if not isinstance(element_repr, string_types):
        raise TypeError("element must be represented by a string-type")
    
    element = element_repr.title().strip().split()[0]
    try:
        index = periodic_table.index(element)

    except IndexError:
        raise ValueError("unrecognized element '{}'".format(element_repr))
    except ValueError:
        try:
            return common_molecule_name2Z[element]
        except KeyError:
            raise ValueError("unrecognized element '{}'".format(element_repr))
        

    return 1 + index
    





def species_to_element(species):
    """ Converts a floating point representation of a species to a string
    representation of the element and its ionization state """
    
    if not isinstance(species, (float, int)):
        raise TypeError("species must be represented by a floating point-type")
    
    if round(species,1) != species:
        # Then you have isotopes, but we will ignore that
        species = int(species*10)/10.

    if species + 1 >= len(periodic_table) or 1 > species:
        # Don"t know what this element is. It"s probably a molecule.
        try:
            elems = common_molecule_species2elems[species]
            return "-".join(elems)
        except KeyError:
            # No idea
            return str(species)
        
    atomic_number = int(species)
    element = periodic_table[int(species) - 1]
    ionization = int(round(10 * (species - int(species)) + 1))

    # The special cases
    if element in ("C", "H", "He"): return element
    return "%s %s" % (element, "I" * ionization)



def atomic_number_to_element(Z):
    """
    Converts a string representation of an element and its ionization state
    to a floating point.

    :param element_repr:
        A string representation of the element. Typical examples might be 'Fe',
        'Ti I', 'si'.
    """
    
    elem = species_to_element(float(Z))
    return elem.split()[0]

def elems_isotopes_ion_to_species(elem1,elem2,isotope1,isotope2,ion):
    Z1 = int(element_to_species(elem1.strip()))
    if isotope1==0: isotope1=''
    else: isotope1 = str(isotope1).zfill(2)

    if elem2.strip()=='': # Atom
        mystr = "{}.{}{}".format(Z1,int(ion-1),isotope1)
    else: # Molecule
        #assert ion==1,ion
        Z2 = int(element_to_species(elem2.strip()))

        # If one isotope is specified but the other isn't, use a default mass
        # These masses are taken from MOOG for Z=1 to 95
        amu = [1.008,4.003,6.941,9.012,10.81,12.01,14.01,16.00,19.00,20.18,
               22.99,24.31,26.98,28.08,30.97,32.06,35.45,39.95,39.10,40.08,
               44.96,47.90,50.94,52.00,54.94,55.85,58.93,58.71,63.55,65.37,
               69.72,72.59,74.92,78.96,79.90,83.80,85.47,87.62,88.91,91.22,
               92.91,95.94,98.91,101.1,102.9,106.4,107.9,112.4,114.8,118.7,
               121.8,127.6,126.9,131.3,132.9,137.3,138.9,140.1,140.9,144.2,
               145.0,150.4,152.0,157.3,158.9,162.5,164.9,167.3,168.9,173.0,
               175.0,178.5,181.0,183.9,186.2,190.2,192.2,195.1,197.0,200.6,
               204.4,207.2,209.0,210.0,210.0,222.0,223.0,226.0,227.0,232.0,
               231.0,238.0,237.0,244.0,243.0]
        amu = [int(round(x,0)) for x in amu]
        if isotope1 == '':
            if isotope2 == 0:
                isotope2 = ''
            else:
                isotope1 = str(amu[Z1-1]).zfill(2)
        else:
            if isotope2 == 0:
                isotope2 = str(amu[Z2-1]).zfill(2)
            else:
                isotope2 = str(isotope2).zfill(2)
        # Swap if needed
        if Z1 < Z2:
            mystr = "{}{:02}.{}{}{}".format(Z1,Z2,int(ion-1),isotope1,isotope2)
        else:
            mystr = "{}{:02}.{}{}{}".format(Z2,Z1,int(ion-1),isotope2,isotope1)

    return float(mystr)

def species_to_elems_isotopes_ion(species):
    element = species_to_element(species)
    if species >= 100:
        # Molecule
        Z1 = int(species/100)
        Z2 = int(species - Z1*100)
        elem1 = species_to_element(Z1).split()[0]
        elem2 = species_to_element(Z2).split()[0]
        # All molecules that we use are unionized
        ion = 1
        if species == round(species,1):
            # No isotope specified
            isotope1 = 0
            isotope2 = 0
        else: #Both isotopes need to be specified!
            isotope1 = int(species*1000) - int(species*10)*100
            isotope2 = int(species*100000) - int(species*1000)*100
            if isotope1 == 0 or isotope2 == 0: 
                raise ValueError("molecule species must have both isotopes specified: {} -> {} {}".format(species,isotope1,isotope2))
        # Swap if needed
    else:
        # Element
        try:
            elem1,_ion = element.split()
        except ValueError as e:
            if element == 'C':
                elem1,_ion = 'C','I'
            elif element == 'H':
                elem1,_ion = 'H','I'
            elif element == 'He':
                elem1,_ion = 'He','I'
            else:
                print(element)
                raise e
        ion = len(_ion)
        assert _ion == 'I'*ion, "{}; {}".format(_ion,ion)
        if species == round(species,1):
            isotope1 = 0
        elif species == round(species,4):
            isotope1 = int(species*10000) - int(species*10)*1000
        elif species == round(species,3):
            isotope1 = int(species*1000) - int(species*10)*100
        else:
            raise ValueError("problem determining isotope: {}".format(species))
        elem2 = ''
        isotope2 = 0
    return elem1,elem2,isotope1,isotope2,ion


def get_common_letters(strlist):
    return "".join([x[0] for x in zip(*strlist) \
        if reduce(lambda a,b:(a == b) and a or None,x)])


def find_common_start(strlist):
    strlist = strlist[:]
    prev = None
    while True:
        common = get_common_letters(strlist)
        if common == prev:
            break
        strlist.append(common)
        prev = common

    return get_common_letters(strlist)


def extend_limits(values, fraction=0.10, tolerance=1e-2):
    """ Extend the values of a list by a fractional amount """

    values = np.array(values)
    finite_indices = np.isfinite(values)

    if np.sum(finite_indices) == 0:
        raise ValueError("no finite values provided")

    lower_limit, upper_limit = np.min(values[finite_indices]), np.max(values[finite_indices])
    ptp_value = np.ptp([lower_limit, upper_limit])

    new_limits = lower_limit - fraction * ptp_value, ptp_value * fraction + upper_limit

    if np.abs(new_limits[0] - new_limits[1]) < tolerance:
        if np.abs(new_limits[0]) < tolerance:
            # Arbitrary limits, since we"ve just been passed zeros
            offset = 1

        else:
            offset = np.abs(new_limits[0]) * fraction
            
        new_limits = new_limits[0] - offset, offset + new_limits[0]

    return np.array(new_limits)


