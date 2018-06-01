# coding: utf-8

""" Misc utility functions """

from six import string_types

def vac2air(lamvac):
    """
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Morton 2000
    """
    s2 = (1e4/lamvac)**2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return lamvac/n

def air2vac(lamair):
    """
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Piskunov
    """
    s2 = (1e4/lamair)**2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2)
    return lamair*n
