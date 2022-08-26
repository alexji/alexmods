from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from astropy import table
from astropy import coordinates as coord
from astropy import units as u

import warnings

from six import string_types

import os
basepath = os.path.dirname(__file__)
datapath = os.path.join(basepath,"data")

# DataFrame subclass
#from . import gaiatools as gtool
#from pyia import GaiaData

from .smhutils import element_to_species,species_to_element,element_to_atomic_number
#from smh.photospheres.abundances import asplund_2009 as solar_composition

## Used to verify periodic table
from .periodic import element as PTelement

## Regular expressions
import re
m_XH = re.compile('\[(\D+)/H\]')
m_XFe= re.compile('\[(\D+)/Fe\]')

######################################
# Solar Composition                  #
######################################
# used meteoritic if photospheric not available
_asplund09 = {'H':12.0,'He':10.93,'Li':1.05,'Be':1.38,'B':2.70,
              'C' :8.43,'N':7.83,'O':8.69,'F':4.56,'Ne':7.93,
              'Na':6.24,'Mg':7.60,'Al':6.45,'Si':7.51,'P':5.41,
              'S':7.12,'Cl':5.50,'Ar':6.40,'K' :5.03,'Ca':6.34,
              'Sc':3.15,'Ti':4.95,'V' :3.93,'Cr':5.64,'Mn':5.43,
              'Fe':7.50,'Co':4.99,'Ni':6.22,'Cu':4.19,'Zn':4.56,
              'Ga':3.04,'Ge':3.65,'As':2.30,'Se':3.34,'Br':2.54,
              'Kr':3.25,'Rb':2.52,'Sr':2.87,'Y' :2.21,'Zr':2.58,
              'Nb':1.46,'Mo':1.88,'Ru':1.75,'Rh':0.91,'Pd':1.57,
              'Ag':0.94,'Cd':1.71,'In':0.80,'Sn':2.04,'Sb':1.01,
              'Te':2.18, 'I':1.55,'Xe':2.24,'Cs':1.08,
              'Ba':2.18,'La':1.10,'Ce':1.58,'Pr':0.72,'Nd':1.42,
              'Sm':0.96,'Eu':0.52,'Gd':1.07,'Tb':0.30,
              'Dy':1.10,'Ho':0.48,'Er':0.92,'Tm':0.10,'Yb':0.84,
              'Lu':0.10,'Hf':0.85,'Ta':-0.12,'W':0.85,'Re':0.26,
              'Os':1.40,'Ir':1.38,'Pt':1.62, 'Au':0.92,'Hg':1.17,
              'Tl':0.90,'Pb':1.75,'Bi':0.65,'Th':0.02,'U':-0.54,
              'Tc':np.nan}
def get_solar(elems):
    elems = np.ravel(elems)
    good_elems = [getelem(elem) for elem in elems]
    return pd.Series([_asplund09[elem] for elem in good_elems],index=elems,name='asplund09')

######################################
# Utility functions for column names #
######################################
def epscolnames(df):
    return _getcolnames(df,'eps')
def errcolnames(df):
    return _getcolnames(df,'e_')
def ulcolnames(df):
    return _getcolnames(df,'ul')
def XHcolnames(df):
    return _getcolnames(df,'XH')
def XFecolnames(df):
    return _getcolnames(df,'XFe')

def epscol(elem):
    return 'eps'+getelem(elem,lower=True)
def errcol(elem):
    try:
        return 'e_'+getelem(elem,lower=True)
    except ValueError:
        if elem=="alpha": return "e_alpha"
        else: raise
def eABcol(elems):
    A,B = elems
    return f"eAB_{getelem(A)}/{getelem(B)}"
def ulcol(elem):
    try:
        return 'ul'+getelem(elem,lower=True)
    except ValueError:
        if elem=="alpha": return "ulalpha"
        else: raise
def XHcol(elem,keep_species=False):
    try:
        return '['+getelem(elem,keep_species=keep_species)+'/H]'
    except ValueError:
        if elem=="alpha": return "[alpha/H]"
        else: raise
def XFecol(elem,keep_species=False):
    try:
        return '['+getelem(elem,keep_species=keep_species)+'/Fe]'
    except ValueError:
        if elem=="alpha": return "[alpha/Fe]"
        else: raise
def ABcol(elems):
    """ Note: by default the data does not have [A/B] """
    A,B = elems
    return '['+getelem(A)+'/'+getelem(B)+']'
def make_XHcol(species):
    if species==22.0: return "[Ti I/H]"
    if species==23.1: return "[V II/H]"
    if species==26.1: return "[Fe II/H]"
    if species==24.1: return "[Cr II/H]"
    if species==38.0: return "[Sr I/H]"
    if species==106.0: return "[C/H]"
    if species==607.0: return "[N/H]"
    return XHcol(species)
def make_XFecol(species):
    if species==22.0: return "[Ti I/Fe]"
    if species==23.1: return "[V II/Fe]"
    if species==26.1: return "[Fe II/Fe]"
    if species==24.1: return "[Cr II/Fe]"
    if species==38.0: return "[Sr I/Fe]"
    if species==106.0: return "[C/Fe]"
    if species==607.0: return "[N/Fe]"
    return XFecol(species)
def make_epscol(species):
    if species==22.0: return "epsti1"
    if species==23.1: return "epsv2"
    if species==26.1: return "epsfe2"
    if species==24.1: return "epscr2"
    if species==38.0: return "epssr1"
    if species==106.0: return "epsc"
    if species==607.0: return "epsn"
    return epscol(species)
def make_errcol(species):
    if species==22.0: return "e_ti1"
    if species==23.1: return "e_v2"
    if species==26.1: return "e_fe2"
    if species==24.1: return "e_cr2"
    if species==38.0: return "e_sr1"
    if species==106.0: return "e_c"
    if species==607.0: return "e_n"
    return errcol(species)
def make_ulcol(species):
    if species==22.0: return "ulti1"
    if species==23.1: return "ulv2"
    if species==26.1: return "ulfe2"
    if species==24.1: return "ulcr2"
    if species==38.0: return "ulsr1"
    if species==106.0: return "ulc"
    if species==607.0: return "uln"
    return ulcol(species)

def _getcolnames(df,prefix):
    allnames = []
    for col in df:
        try:
            this_prefix,elem = identify_prefix(col)
        except ValueError:
            continue
        if this_prefix==prefix: allnames.append(col)
    return allnames

def getelem(elem,lower=False,keep_species=False):
    """
    Converts common element names into a formatted symbol
    """
    common_molecules = {'CH':'C','NH':'N'}
    special_ions = ['Ti I','Cr II']
    if isinstance(elem, string_types):
        prefix = None
        try:
            prefix,elem_ = identify_prefix(elem)
            elem = elem_
        except ValueError:
            pass

        if PTelement(elem) != None: # No ionization, e.g. Ti
            elem = PTelement(elem).symbol
        elif elem in common_molecules:
            elem = common_molecules[elem]
        elif prefix != None and '.' in elem:
            elem,ion = elem.split('.')
            elem = format_elemstr(elem)
        #elif '.' in elem: #Not sure if this works correctly yet
        #    elem,ion = elem.split('.')
        #    elem = format_elemstr(elem)
        elif elem[-1]=='I': #Check for ionization
            # TODO account for ionization
            if ' ' in elem: #of the form 'Ti II' or 'Y I'
                species = element_to_species(elem)
                elem = species_to_element(species)
                elem = elem.split()[0]
            else: #of the form 'TiII' or 'YI'
                if elem[0]=='I':
                    assert elem=='I'*len(elem)
                    elem = 'I'
                else:
                    while elem[-1] == 'I': elem = elem[:-1]
        else:
            # Use smh to check for whether element is in periodic table
            species = element_to_species(elem)
            elem = species_to_element(species)
            elem = elem.split()[0]
    elif isinstance(elem, (int, np.integer)):
        elem = int(elem)
        elem = PTelement(elem)
        ## TODO common molecules
        assert elem != None
        elem = elem.symbol
        if keep_species: raise NotImplementedError()
    elif isinstance(elem, float):
        species = elem
        elem = species_to_element(species)
        if not keep_species: elem = elem.split()[0]

    if lower: elem = elem.lower()
    return elem

def format_elemstr(elem):
    assert len(elem) <= 2 and len(elem) >= 1
    return elem[0].upper() + elem[1:].lower()
def getcolion(col):
    prefix,elem = identify_prefix(col)
    if '.' in elem: int(ion = elem.split('.')[1])
    else: ion = get_default_ion(elem)
    ionstr = 'I'
    for i in range(ion): ionstr += 'I'
    return ionstr
def identify_prefix(col):
    for prefix in ['eps','e_','ul','XH','XFe']:
        if prefix in col:
            return prefix, col[len(prefix):]
        if prefix=='XH':
            matches = m_XH.findall(col)
            if len(matches)==1: return prefix,matches[0]
        if prefix=='XFe':
            matches = m_XFe.findall(col)
            if len(matches)==1: return prefix,matches[0]
    raise ValueError("Invalid column:"+str(col))
def get_default_ion(elem):
    default_to_1 = ['Na','Mg','Al','Si','Ca','Cr','Mn','Fe','Co','Ni']
    default_to_2 = ['Sc','Ti','Sr','Y','Zr','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Dy']
    elem = getelem(elem)
    if elem in default_to_1:
        return 1
    elif elem in default_to_2:
        return 2
    else:
        warnings.warn("get_default_ion: {} not in defaults, returning 2".format(elem))
        return 2

########################################################
# Utility functions operating on standard DataFrame(s) #
########################################################
def get_star_abunds(starname,data,type):
    """
    Take a single row, pull out the columns of a specific type ('eps', 'XH', 'XFe', 'e_', 'ul'), give simple element names
    """
    assert type in ['eps','XH','XFe','e_','ul']
    star = data.ix[starname]
    colnames = _getcolnames(data,type)
    if len(colnames)==0: raise ValueError("{} not in data".format(type))
    abunds = np.array(star[colnames])
    elems = [getelem(elem) for elem in colnames]
    return pd.Series(abunds,index=elems)

def XH_from_eps(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        asplund = get_solar(epscols)
        for col in epscols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] - float(asplund[col])
def XFe_from_eps(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        assert 'epsfe' in epscols
        asplund = get_solar(epscols)
        feh = df['epsfe'] - float(asplund['epsfe'])
        for col in epscols:
            if col=='epsfe': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            XH = df[col]-float(asplund[col])
            df[XFecol(col)] = XH - feh
def eps_from_XH(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        asplund = get_solar(XHcols)
        for col in XHcols:
            if epscol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(epscol(col)))
            df[epscol(col)] = df[col] + float(asplund[col])
def XFe_from_XH(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        assert '[Fe/H]' in XHcols
        feh = df['[Fe/H]']
        for col in XHcols:
            if col=='[Fe/H]': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            df[XFecol(col)] = df[col] - feh
def eps_from_XFe(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            df[epscol(col)] = df[col] + feh + float(asplund[col])
def XH_from_XFe(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] + feh

#########################
# Load halo data tables #
#########################
def load_jinabase(key=None,p=1,load_eps=True,load_ul=True,load_XH=True,load_XFe=True,load_aux=True,
                  name_as_index=False):
    """
    Load our local copy of jinabase (http://jinabase.pythonanywhere.com/, Abohalima and Frebel 2018)
    """
    ## Read data
    data = pd.read_csv(datapath+"/abundance_tables/jinabase_all_info.txt", header=0, delim_whitespace=True, index_col=0,
                       na_values=["*"])
    uls  = pd.read_csv(datapath+"/abundance_tables/jinabase_limits_key.txt", header=0, delim_whitespace=True, index_col=0,
                       na_values=["*"])#true_values=['1.00','4.00'],false_values=['*'])
    uls = pd.notnull(uls)
    
    ## remove duplicate entries
    if p==1 or p==2:
        ii = data["key"]==p
        data = data[ii]
        uls = uls[ii]
    Nstars = len(data)

    ## Ca, Ti, V, Cr, Mn, and Fe have II in the table
    ## We will simply drop those columns
    ## TODO we probably want to have Ti II instead of Ti I
    print("WARNING: dropping CaII, TiII, VII, CrII, MnII, FeII columns")
    
    elems = data.columns[24:-6]
    uls = uls[uls.columns[4:-6]]
    epscolnames = list(map(lambda x: "eps"+x.lower(), elems))
    data.rename(columns=dict(zip(elems,epscolnames)), inplace=True)
    
    ## save auxiliary data
    aux_cols = data.columns[1:24]
    auxdata = data[aux_cols]
    
    ## Create ul data
    ulelems = 'ul' + elems
    data = data[epscolnames]
    ultab = uls[ulelems]
    
    if not load_ul:
        # Remove ul
        dmat = data.as_matrix()
        ul_flag = pd.notnull(uls).as_matrix()
        dmat[ul_flag] = np.nan
        data = pd.DataFrame(dmat, columns=epscolnames, index=data.index)
    else:
        ultab.columns = list(map(lambda x: x.lower(), ultab.columns))
        data = pd.concat([data,ultab],axis=1)
    if key != None:
        assert key in np.unique(auxdata["Sci_key"]), np.unique(auxdata["Sci_key"])
        data = data[auxdata["Sci_key"]==key]
        auxdata = auxdata[auxdata["Sci_key"]==key]
    else:
        assert len(data) == Nstars, (len(data),Nstars)

    if load_XH:
        XH_from_eps(data)
    if load_XFe:
        XFe_from_eps(data)
    if load_aux:
        data = pd.concat([data,auxdata],axis=1)
    else:
        data = pd.concat([data,auxdata[["Name","Reference"]]],axis=1)
    if not load_eps:
        data.drop(epscolnames,axis=1,inplace=True)
    if name_as_index:
        data.index = data["Name"]
    return data
def load_halo(**kwargs):
    """ load_jinabase, remove stars with loc DW and UF """
    halo = load_jinabase(**kwargs)
    halo = halo[halo["Loc"] != "DW"]
    halo = halo[halo["Loc"] != "UF"]
    return halo
def load_cldw(add_all=False, **kwargs):
    cldw = load_jinabase(**kwargs)
    cldw = cldw[cldw["Loc"] == "DW"]
    def get_gal(row):
        ## These are papers from a single galaxy
        refgalmap = {"AOK07b":"UMi","COH10":"UMi","URA15":"UMi",
                     "FRE10a":"Scl","GEI05":"Scl","JAB15":"Scl","SIM15":"Scl","SKU15":"Scl",
                     "AOK09":"Sex",
                     "FUL04":"Dra","COH09":"Dra","TSU15":"Dra","TSU17":"Dra",
                     "NOR17":"Car","VEN12":"Car",
                     "HAN18":"Sgr"}
        ref = row["Reference"]
        if ref in refgalmap:
            return refgalmap[ref]
        ## These are papers with multiple galaxies
        assert ref in ["SHE01","SHE03","TAF10","KIR12"], ref
        name = row["Name"]
        name = name[0].upper() + name[1:3].lower()
        if name == "Umi": return "UMi"
        return name
    #allrefs = np.unique(cldw["Reference"])
    #multirefs = ["SHE01","SHE03","TAF10","KIR12"]
    gals = [get_gal(x) for i,x in cldw.iterrows()]
    cldw["galaxy"] = gals

    if add_all:
        fnx = load_letarte10_fornax()
        scl = load_hill19_sculptor()
        car = load_lemasle12_carina()
        sgr = load_apogee_sgr()
        cldw = pd.concat([cldw,fnx,scl,car,sgr],axis=0)
    return cldw

def load_roed(match_anna_elems=True,load_eps=True,load_ul=True,load_XH=True,load_XFe=True):
    """ Load from Ian's 2014 table (TODO: add stellar parameters etc) """
    roed = ascii.read(datapath+'/abundance_tables/roederer14.tsv',data_start=3)
    roed = roed.to_pandas()
    
    ions = np.unique(roed['Ion'])
    a2r_map = {}
    picked_ions = {'epsc':'C (CH)',
                   'epsn':'N (NH)',
                   'epsfe':'Fe I',
                   'epscr':'Cr I',
                   'epsmn':'Mn I',
                   'epsti':'Ti II',
                   'epsv':'V II'}

    epscols = np.unique(list(map(lambda x: 'eps'+x.split()[0].lower(), ions)))

    for fullion in ions:
        elem,ion = fullion.split()
        a = 'eps'+elem.lower()
        if a in epscols:
            if a in picked_ions:
                a2r_map[a] = picked_ions[a]
            else:
                a2r_map[a] = fullion
        else:
            print("skipping",a)
    elem_groups = roed.groupby('Ion')
    
    roed_eps = []
    roed_err = []
    roed_ul  = []
    for col in epscols:
        if col in a2r_map:
            rcol = a2r_map[col]
            rdf = elem_groups.get_group(rcol)
            eps = pd.Series(rdf['log(e)'],name=epscol(col),copy=True)
            eps.index = rdf['Name']
            # skip s_[X/Fe], e2_[X/Fe], e3_[X/Fe]
            err = pd.Series(rdf['e_[X/Fe]'],name=errcol(col),copy=True)
            err.index = rdf['Name']
            ul  = pd.Series(rdf['l_log(e)']=='<',name=ulcol(col),copy=True)
            ul.index = rdf['Name']
            roed_eps.append(eps)
            roed_err.append(err)
            roed_ul.append(ul)
        else:
            print(col,'not in roed')
    #epstab = pd.DataFrame(roed_eps)
    #errtab = pd.DataFrame(roed_err)
    #ultab  = pd.DataFrame(roed_ul)
    #rdf = pd.concat([epstab,errtab,ultab]).transpose()
    #return rdf

    epstab = pd.DataFrame(roed_eps).transpose()
    errtab = pd.DataFrame(roed_err).transpose()
    ultab  = pd.DataFrame(roed_ul).transpose()
    for col in ultab.columns: # Indexing with NaN is bad
        ultab[col][pd.isnull(ultab[col])] = False
    
    all_elems = []
    for col in epstab:
        if 'eps' in col: 
            all_elems.append(col[3:])

    if not load_ul:
        raise NotImplementedError #broken right now
        # Remove upper limits from table
        for elem in all_elems:
            try:
                epstab['eps'+elem][ultab['ul'+elem]] = np.nan
            except KeyError: continue

    data = epstab
    if load_ul:
        data = pd.merge(data,ultab,how='inner',left_index=True,right_index=True)
        # Should eventually figure out how to do this without deleting the extra columns...
        #data.drop(['p_y','name_y','simbad_y','ref_y'],axis=1,inplace=True)
        data.rename(columns=lambda x: x if '_x' not in x else x[:-2],inplace=True)
    if load_XH:
        XH_from_eps(data)
    if load_XFe: 
        XFe_from_eps(data)
    if not load_eps:
        data.drop(['eps'+elem for elem in all_elems],axis=1,inplace=True)
    return data
def load_yong():
    """
    This is a stub
    """
    df6 = ascii.read(datapath+"/abundance_tables/yongtab6.dat").to_pandas()
    df7 = ascii.read(datapath+"/abundance_tables/yongtab7.dat").to_pandas()
    df = pd.concat([df6,df7],axis=1)
    return df

#########################
# Load Alex's UFD table #
#########################
def parse_dwarf_table(alpha=False,table_path=datapath+'/abundance_tables/dwarf_lit_all.tab'):
    def _process_column(d,elem):
        N = len(d)
        if elem not in d.colnames:
            raise ValueError(elem)
        old_col = d[elem]
        ul_col = table.MaskedColumn(data=np.zeros(N),name='ul_'+elem,dtype=int)
        if old_col.dtype==np.float:
            return old_col,ul_col
    
        new_col = table.MaskedColumn(data=np.zeros(N,dtype=np.float),name=elem)
        for i in range(N):
            if np.ma.is_masked(old_col[i]):
                new_col[i]      = np.ma.masked
                ul_col[i]       = np.ma.masked
            elif old_col[i][0]=='<':
                new_col[i] = np.float(old_col[i][1:])
                ul_col[i] = 1
            else:
                new_col[i] = np.float(old_col[i])
        return new_col,ul_col
    d = table.Table(ascii.read(table_path))
    elems = d.colnames[2:-4]
    for elem in elems:
        elem_col,ul_col = _process_column(d,elem)
        d.remove_column(elem)
        d.add_column(elem_col)
        d.add_column(ul_col)
    if alpha:
        col = table.MaskedColumn(np.nanmean([d['Mg'],d['CaI'],d['TiII']],axis=0),name='alpha')
        d.add_column(col)
        col = table.MaskedColumn(np.zeros(len(d)),name='ul_alpha',dtype=int)
        d.add_column(col)
    return d
def load_ufds(load_all=False,load_eps=True,load_ul=True,load_XH=True,load_XFe=True,alpha=False):
    ufds = parse_dwarf_table(alpha=alpha)
    ufds = ufds.to_pandas()
    ufds.index = ufds['Star']
    def column_renamer(x):
        if x=='Source': return 'ref'
        elif x in ["alpha","ul_alpha"]: return x
        elif 'ul_' in x: return ulcol(getelem(x[3:]))
        else:
            try:
                elem = getelem(x)
                if elem=='Fe': return XHcol(elem)
                return XFecol(elem)
            except:
                return x.lower()
    ufds.rename(columns=column_renamer,inplace=True)
    ulcols = ulcolnames(ufds)
    XFecols = XFecolnames(ufds)
    if not load_ul: 
        #remove data with upper limits
        for col in XFecols:
            upper_limits = np.array(ufds[ulcol(col)]==1)
            ufds[col][upper_limits] = np.nan #raises warning, seems to be ok though
        ufds.drop(ulcols,axis=1,inplace=True)
    else:
        for col in ulcols:
            ufds[col] = ufds[col]==1
            ufds[col][pd.isnull(ufds[col])] = False
    if load_eps: eps_from_XFe(ufds)
    if load_XH:  XH_from_XFe(ufds)
    if not load_XFe: ufds.drop(XFecols,axis=1,inplace=True)
    return ufds

##################
# Load r-process #
##################
def load_bisterzo(model="B14"):
    assert model in ["T04", "B14"], model
    if model=="B14":
        dcol = "TW+"
        ecol = "e_TW+"
    if model=="T04":
        dcol = "T04+"
        ecol = "e_T04+"
    path = datapath+"/rproc_patterns/"
    tab = ascii.read(path+"bisterzo14/table1.dat", readme=path+"bisterzo14/ReadMe")
    tab = tab[np.array(list(map(lambda x: "^" not in x, tab["Isotope"])))]

    solar_total = get_solar(list(tab["Isotope"])+["Th","U"])
    f_s = pd.Series(np.array(tab[dcol]), index=np.array(tab["Isotope"]))*.01
    f_s.loc["Th"] = 0.0; f_s.loc["U"] = 0.0
    f_r = 1.0 - f_s
    spat = solar_total + np.log10(f_s)
    rpat = solar_total + np.log10(f_r)
    return rpat, spat
def load_arlandini(model="stellar"):
    """ This is my manually processed table """
    import plot_ncap_pattern as pnp
    assert model in ["classical", "stellar"], model
    if model == "classical":
        Nrcol = "Nr2"
    elif model == "stellar":
        Nrcol = "Nr1"
    df = ascii.read(datapath+"/rproc_patterns/arlandini99.txt").to_pandas()
    dfsum = df.groupby("El").sum()
    rarr = dfsum[Nrcol]
    Zarr = pnp.get_Zarr(rarr)
    iisort = np.argsort(Zarr)
    rpat = np.log10(rarr[iisort])
    return rpat
def load_arlandini():
    df = ascii.read(datapath+"/rproc_patterns/arlandini99.txt").to_pandas()
    df["Z"] = list(map(lambda x: int(element_to_species(x)), df["El"]))
    df.index = zip(df["Z"].values.astype(int), df["A"].values.astype(int))
    return df
def load_lodders():
    lodders = ascii.read(datapath+"/rproc_patterns/lodders10_isotopes.txt").to_pandas()
    lodders.index = zip(lodders["Z"].values.astype(int), lodders["A"].values.astype(int))
    return lodders    
def load_arnould():
    df = ascii.read(datapath+"/rproc_patterns/r_process_arnould_2007.txt",delimiter="&").to_pandas()
    A = df["A"]
    df["Z"] = list(map(lambda x: int(element_to_species(x)), df["Elem"]))
    df.index = zip(df["Z"].values.astype(int), df["A"].values.astype(int))
    return df    

#######################
# Load NearbyGalaxies #
#######################
def load_galdata():
    fname = datapath+"/mcconnachie_plus/NearbyGalaxies.dat"
    
    names = ["Galaxy","RA","Dec","EB-V","m-M","m-M_e1","m-M_e2","vh","vh_e1","vh_e2",
             "Vmag","Vmag_e1","Vmag_e2","PA","PA_e1","PA_e2","ell","ell_e1","ell_e2",
             "muVo","muVo_e1","muVo_e2","rh","rh_e1","rh_e2","sigma_s","sigma_s_e1","sigma_s_e2",
             "vrot_s","vrot_s_e1","vrot_s_e2","MHI","sigma_g","sigma_g_e1","sigma_g_e2",
             "vrot_g","vrot_g_e1","vrot_g_e2","[Fe/H]","[Fe/H]_e1","[Fe/H]_e2","F",
             "gal", "References"]
    col_starts = [0,19,30,40,46,52,57,62,69,74,79,84,88,92,98,103,108,113,118,123,128,132,136,143,149,155,160,165,170,175,180,185,190,195,200,205,211,216,221,227,232,237,239,257]
    col_ends   = [x-1 for x in col_starts[1:]] + [280]
    tab = ascii.read(fname,
                     format='fixed_width_no_header',
                     names=names,
                     col_starts=col_starts,
                     col_ends  =col_ends)
    good_gals = np.array(list(map(lambda x: False if x[0]=="*" else True, tab["Galaxy"])))
    gals_to_remove = ["LMC","SMC","Pisces II","Willman 1","Sagittarius dSph"]
    good_gals = np.logical_and(good_gals, list(map(lambda x: False if x in gals_to_remove else True, tab["Galaxy"])))
    good = tab[good_gals]
    df = good.to_pandas()
    #e1 is plus error, e2 is minus error
    df["VMag"] = df["Vmag"] - df["m-M"]
    df["VMag_e1"] = df["Vmag_e1"] + df["m-M_e1"]
    df["VMag_e2"] = df["Vmag_e2"] + df["m-M_e2"]
    df["logLsun"] = ((df["VMag"]-4.8)/-2.5)+.012
    df["distance"] = 0.01 * 10**(df["m-M"]/5.)
    df["distance_e1"] = 0.01 * 10**((df["m-M"]+df["m-M_e2"])/5.) - df["distance"]
    df["distance_e2"] = df["distance"] - 0.01 * 10**((df["m-M"]-df["m-M_e1"])/5.)
    df["rh_pc"] = 1000.*df["distance"]*(np.array(df["rh"])*u.arcmin).to(u.radian).value
    df["rh_pc_e1"] = 1000.*df["distance_e1"]*(np.array(df["rh"])*u.arcmin).to(u.radian).value + 1000.*df["distance"]*(np.array(df["rh_e1"])*u.arcmin).to(u.radian).value
    df["rh_pc_e2"] = 1000.*df["distance_e2"]*(np.array(df["rh"])*u.arcmin).to(u.radian).value + 1000.*df["distance"]*(np.array(df["rh_e2"])*u.arcmin).to(u.radian).value
    df["Mdyn"] = 580. * df["rh_pc"] * df["sigma_s"]**2 
    df["Mdyn_e1"] = 580. * (df["rh_pc"]+df["rh_pc_e1"]) * (df["sigma_s"]+df["sigma_s_e1"])**2 - df["Mdyn"]
    df["Mdyn_e2"] = df["Mdyn"] - 580. * (df["rh_pc"]-df["rh_pc_e2"]) * (df["sigma_s"]-df["sigma_s_e2"])**2
    return df

def load_galdata_notMW():
    fname = datapath+"/mcconnachie_plus/NearbyGalaxiesNotMW.dat"
    
    names = ["Galaxy","RA","Dec","EB-V","m-M","m-M_e1","m-M_e2","vh","vh_e1","vh_e2",
             "Vmag","Vmag_e1","Vmag_e2","PA","PA_e1","PA_e2","ell","ell_e1","ell_e2",
             "muVo","muVo_e1","muVo_e2","rh","rh_e1","rh_e2","sigma_s","sigma_s_e1","sigma_s_e2",
             "vrot_s","vrot_s_e1","vrot_s_e2","MHI","sigma_g","sigma_g_e1","sigma_g_e2",
             "vrot_g","vrot_g_e1","vrot_g_e2","[Fe/H]","[Fe/H]_e1","[Fe/H]_e2","F",
             "gal", "References"]
    col_starts = [0,19,30,40,46,52,57,62,69,74,79,84,88,92,98,103,108,113,118,123,128,132,136,143,149,155,160,165,170,175,180,185,190,195,200,205,211,216,221,227,232,237,239,257]
    col_ends   = [x-1 for x in col_starts[1:]] + [280]
    tab = ascii.read(fname,
                     format='fixed_width_no_header',
                     names=names,
                     col_starts=col_starts,
                     col_ends  =col_ends)
    good_gals = np.array(list(map(lambda x: False if x[0]=="*" else True, tab["Galaxy"])))
    #gals_to_remove = ["LMC","SMC","Pisces II","Willman 1","Sagittarius dSph"]
    #good_gals = np.logical_and(good_gals, list(map(lambda x: False if x in gals_to_remove else True, tab["Galaxy"])))
    good = tab[good_gals]
    df = good.to_pandas()
    #e1 is plus error, e2 is minus error
    df["VMag"] = df["Vmag"] - df["m-M"]
    df["VMag_e1"] = df["Vmag_e1"] + df["m-M_e1"]
    df["VMag_e2"] = df["Vmag_e2"] + df["m-M_e2"]
    df["logLsun"] = ((df["VMag"]-4.8)/-2.5)+.012
    df["distance"] = 0.01 * 10**(df["m-M"]/5.)
    df["distance_e1"] = 0.01 * 10**((df["m-M"]+df["m-M_e2"])/5.) - df["distance"]
    df["distance_e2"] = df["distance"] - 0.01 * 10**((df["m-M"]-df["m-M_e1"])/5.)
    df["rh_pc"] = 1000.*df["distance"]*(np.array(df["rh"])*u.arcmin).to(u.radian).value
    df["rh_pc_e1"] = 1000.*df["distance_e1"]*(np.array(df["rh"])*u.arcmin).to(u.radian).value + 1000.*df["distance"]*(np.array(df["rh_e1"])*u.arcmin).to(u.radian).value
    df["rh_pc_e2"] = 1000.*df["distance_e2"]*(np.array(df["rh"])*u.arcmin).to(u.radian).value + 1000.*df["distance"]*(np.array(df["rh_e2"])*u.arcmin).to(u.radian).value
    df["Mdyn"] = 580. * df["rh_pc"] * df["sigma_s"]**2 
    df["Mdyn_e1"] = 580. * (df["rh_pc"]+df["rh_pc_e1"]) * (df["sigma_s"]+df["sigma_s_e1"])**2 - df["Mdyn"]
    df["Mdyn_e2"] = df["Mdyn"] - 580. * (df["rh_pc"]-df["rh_pc_e2"]) * (df["sigma_s"]-df["sigma_s_e2"])**2
    return df

def load_gcdata():
    gcdir = datapath+"/globular_clusters"
    df1 = ascii.read(gcdir+"/mwgc1.dat").to_pandas()
    df1.index = df1.pop("ID")
    df2 = ascii.read(gcdir+"/mwgc2.dat").to_pandas()
    df2.index = df2.pop("ID")
    df3 = ascii.read(gcdir+"/mwgc3.dat").to_pandas()
    df3.index = df3.pop("ID")
    df = pd.concat([df1,df2,df3], axis=1)
    df["rc_pc"] = 1000.*df["R_Sun"]*(np.array(df["r_c"])*u.arcmin).to(u.radian).value
    df["rh_pc"] = 1000.*df["R_Sun"]*(np.array(df["r_h"])*u.arcmin).to(u.radian).value
    df["Mdyn"] = 580. * df["rh_pc"] * df["sig_v"]**2
    return df

##################
# Supernova yields
##################
def load_hw10(as_number=True):
    assert as_number, "Did not download the mass tables"
    hw10 = Table.read(datapath+"/yield_tables/HW10.znuc.S4.star.el.fits").to_pandas()
    hw10.rename(inplace=True, columns={
            "mass":"Mass","energy":"Energy","mixing":"Mixing","remnant":"Remnant"})
    #elems = [i+1 for i in list(range(83))+[89,91]]
    elems = [i+1 for i in list(range(83))]
    missing_elems = []
    ## the units of this table are mol/g, divided out the ejecta mass.
    ## 1 amu * 1 mol = 1 g
    ## I want Msun/amu of the total ejecta mass, so need to multiply by the ejecta mass in Msun
    ## Units: mol / g * Msun = Msun / amu
    Mej = hw10["Mass"] - hw10["Remnant"]
    for i in elems:
        if str(i) in hw10.columns:
            hw10.rename(inplace=True, columns={str(i):i})
            hw10[i] = hw10[i]*Mej
        else:
            missing_elems.append(i)
    for i in missing_elems: elems.remove(i)
    hw10["Cut"] = "S4"
    cols = elems + list(hw10.columns[0:3].values) + ["Cut"] #,"Remnant"]
    hw10 = hw10[cols]
    hw10["Mass"] = hw10["Mass"].map(lambda x: round(x, 1))
    hw10["Energy"] = hw10["Energy"].map(lambda x: round(x, 1))
    hw10["Mixing"] = hw10["Mixing"].map(lambda x: round(x, 5))
    
    return hw10
    
def load_hw10_old(as_number=False):
    """
    Load Heger + Woosley 2010 Pop III CCSNe yields
    This just adds all the isotopes into element yields
    These are just the models that are in the ApJ paper, but
    online there are a HUGE number more!
    """
    hw10 = Table.read(datapath+"/yield_tables/HW10.fits").to_pandas()
    elems = np.unique(hw10["El"])
    Z = np.zeros(len(hw10),dtype=int)
    for elem in elems:
        ii = hw10["El"] == elem
        _Z = PTelement(elem.decode('utf-8').strip()).atomic
        Z[ii] = _Z
    hw10["Z"] = Z    
    addcol = "Yield"
    if as_number:
        hw10["N"] = hw10["Yield"]/hw10["A"]
        addcol = "N"
    
    models = hw10.groupby(["Mass","Energy","Cut","Mixing"])
    outputyields = []
    for key,df in models:
        mass = df.groupby("Z").sum()[addcol]
        mass.name = key
        outputyields.append(mass)
    hw10y = pd.DataFrame(outputyields)
    hw10y["Mass"] = list(map(lambda x: round(x[0],1), hw10y.index))
    hw10y["Energy"] = list(map(lambda x: round(x[1],1), hw10y.index))
    hw10y["Cut"] = list(map(lambda x: x[2], hw10y.index))
    hw10y["Mixing"] = list(map(lambda x: round(x[3],5), hw10y.index))
    
    return hw10y

def load_hw02():
    """
    Load Heger + Woosley 2002 Pop III PISN yields
    This just adds all the isotopes into element yields and computes their approximate PISN progenitor mass
    """
    hw02 = ascii.read(datapath+"/yield_tables/hw02.txt").to_pandas()
    def elemsplitter(x):
        if x=="He4": return "He",4
        elem = x[:-2]
        A = int(x[-2:])
        return elem, A
    elems,As = np.array(list(map(elemsplitter, hw02["Ion"]))).T
    hw02["elem"] = elems
    hw02["A"] = As
    hw02["Z"] = list(map(element_to_atomic_number, elems))
    outputyields = []
    for model in hw02.columns:
        if not model.startswith("He"): continue
        mass = hw02[[model,"Z"]].groupby("Z").sum()
        mass.name = model
        outputyields.append(mass)
    hw02y = pd.concat(outputyields,axis=1).transpose()
    HeMass = np.array([int(model[2:]) for model in hw02y.index])
    hw02y["HeMass"] = HeMass
    PISNMass = 24./13. * np.array(HeMass) + 20.
    hw02y["PISNMass"] = PISNMass
    return hw02y

def load_nkt13(as_number=True):
    df = pd.read_csv(datapath+"/yield_tables/nkt13.dat", names=["Zmet","M","E","Mrem","Z","A","y"])
    groups = df.groupby(["Zmet","M","E"])
    #yields = groups.sum()["y"]
    allZmet = np.unique(df["Zmet"]); NZmet = allZmet.size
    allM = np.unique(df["M"]); NM = allM.size
    allE = np.unique(df["E"]); NE = allE.size
    imodel = 0
    df["n"] = df["y"]/df["A"]
    
    allZ = np.unique(df["Z"])
    alloutput = []
    for Zmet in allZmet:
        for M in allM:
            for E in allE:
                try:
                    tdf = groups.get_group((Zmet,M,E))
                except:
                    continue
                if as_number:
                    y = tdf.groupby("Z")["n"].sum()
                else:
                    y = tdf.groupby("Z")["y"].sum()
                s = pd.Series([Zmet,M,E],
                              index=["Zmet","M","E"],
                              name=imodel)
                alloutput.append(s.append(y))
                imodel += 1
    df = pd.DataFrame(alloutput)
    return df

def load_simon_galdata(filename=datapath+"/dwarfdata_082918.txt", update_wrong=True):
    galdata = ascii.read(filename,
                         fill_values=[('-9.999','0','DRA'),('-9.999','0','DDEC'),
                                      ('-9.99','0','DELLIP_LOW'), ('-9.99','0','DELLIP_HIGH'),
                                      ('-999.9','0','VHEL'),('-999.9','0','DVHEL_LOW'),('-999.9','0','DVHEL_HIGH'),
                                      ('-9.9','0','SIGMA'),('-9.9','0','DSIGMA_LOW'),('-9.9','0','DSIGMA_HIGH'),
                                      ('-9.99','0','FEH'),('-9.9','0','DFEH_LOW'),('-9.9','0','DFEH_HIGH')])
    df = galdata.to_pandas()
    df.index = df["SHORTNAME"]
    df["DMOD"] = 5.*np.log10(df["DIST"]) + 10
    df["DDMOD_LOW"] = df["DMOD"]-5.*np.log10(df["DIST"]-df["DDIST_LOW"]) + 10
    df["DDMOD_HIGH"] = 5.*np.log10(df["DIST"]+df["DDIST_HIGH"]) + 10 - df["DMOD"]
    df["RHALF_PC"] = 1000.*df["DIST"]*(np.array(df["RHALF"])*u.arcmin).to(u.radian).value
    df["DRHALF_PC_HIGH"] = 1000.*df["DDIST_HIGH"]*(np.array(df["RHALF"])*u.arcmin).to(u.radian).value + 1000.*df["DIST"]*(np.array(df["DRHALF_HIGH"])*u.arcmin).to(u.radian).value
    df["DRHALF_PC_LOW"] = 1000.*df["DDIST_LOW"]*(np.array(df["RHALF"])*u.arcmin).to(u.radian).value + 1000.*df["DIST"]*(np.array(df["DRHALF_LOW"])*u.arcmin).to(u.radian).value
    df["MDYN"] = 580. * df["RHALF_PC"] * df["SIGMA"]**2 # wolf10
    df["DMDYN_HIGH"] = 580. * (df["RHALF_PC"]+df["DRHALF_PC_HIGH"]) * (df["SIGMA"]+df["DSIGMA_HIGH"])**2 - df["MDYN"]
    df["DMDYN_LOW"] = df["MDYN"] - 580. * (df["RHALF_PC"]-df["DRHALF_PC_LOW"]) * (df["SIGMA"]-df["DSIGMA_LOW"])**2
    df["LOGMSTAR"] = (df["M_V"]-4.83)/-2.5 + np.log10(2) # estimating M/L = 2

    if update_wrong:
        print("Updating values that are known to be wrong in literature")
        df.loc["Gru I", "FEH"] = -2.5
    return df

def load_ezzeddine20(filename=datapath+"/abundance_tables/ezzeddine20.txt"):
    df = ascii.read(filename).to_pandas()
    return df

def load_sakari18(filename=datapath+"/abundance_tables/sakari18_merged.txt"):
    df = ascii.read(filename).to_pandas()
    return df
    
def load_holmbeck20(filename=datapath+"/abundance_tables/holmbeck20.txt"):
    df = ascii.read(filename).to_pandas()
    return df
    
def load_hansen18(filename=datapath+"/abundance_tables/hansen18.txt"):
    df = ascii.read(filename).to_pandas()
    return df
    
def load_parsec_isochrones(system="DECAM"):
    #assert system=="DECAM" # for now
    coldict = {
        "DECAM":["umag","gmag","rmag","imag","zmag","Ymag"],
        "WFC3":["F218W1mag","F225W1mag","F275W1mag","F336Wmag","F390Wmag","F438Wmag",
                "F475Wmag","F555Wmag","F606Wmag","F625Wmag","F775Wmag","F814Wmag",
                "F105Wmag","F110Wmag","F125Wmag","F140Wmag","F160Wmag"]
    }
    fname = datapath+"/isochrones/parsec_{}.dat".format(system)
    if not os.path.exists(fname):
        import glob
        fnames = glob.glob(datapath+"/isochrones/parsec*")
        raise ValueError("System {} not in fnames: {}".format(system, fnames))
    cols = ["Zini","MH","logAge","Mini","int_IMF","Mass","logL","logTe","logg","label","mbolmag"] + coldict[system]
    tab = ascii.read(fname, names=cols)
    isodict = {}
    ages = np.unique(tab["logAge"])
    Zs = np.unique(tab["Zini"])
    Zsol = .01471
    #print("Unique Ages:",ages)
    #print("Unique Zs:",Zs)
    for logage in ages:
        for Z in Zs:
            selection = (tab["logAge"]==logage) & (tab["Zini"]==Z)
            age = np.round(10**(logage-9),1)
            logZsun = np.round(np.log10(Z/Zsol),1)
            if np.sum(selection)==0:
                print("Skipping age={} Z={}".format(age,logZsun))
            else:
                isodict[(age,logZsun)] = tab[selection]
    return isodict
def load_dartmouth_isochrones(MH,alpha="ap4",system="DECAM"):
    if system=="DECAM":
        assert MH in [-2.5, -2.0, -1.5, -1.0, -0.5], MH
    else:
        assert MH in [-2.5, -2.0, -1.5], MH
    assert alpha in ["ap0","ap4"]
    coldict = {
        "DECAM":["{}mag".format(x) for x in ["u","g","r","i","z","Y"]],
        "UBVRIJHK":["{}mag".format(x) for x in ["U","B","V","R","I","J","H","Ks","Kp","D51"]],
        "SDSS":["{}mag".format(x) for x in ["u","g","r","i","z"]]
    }
    def make_str(MH):
        if MH==-0.5: return "-05"
        return "{}".format(int(10*MH))
    fname = datapath+"/isochrones/dartmouth_{}_MH{}{}.iso".format(system,make_str(MH),alpha)
    if not os.path.exists(fname):
        import glob
        fnames = glob.glob(datapath+"/isochrones/dartmouth*")
        print(fname)
        raise ValueError("System {} MH {} alpha {} not in fnames: {}".format(system, MH, alpha, fnames))
    cols = ["EEP","Mini","logTe","logg","logL"] + coldict[system] + ["age"]
    tab = ascii.read(fname, names=cols)
    isodict = {}
    logZsun = MH
    for age in np.unique(tab["age"]):
        selection = tab["age"]==age
        isodict[(age,logZsun)] = tab[selection]
    return isodict

def load_pritzl():
    df = ascii.read(datapath+"/abundance_tables/J_AJ_130_2140/table2.dat", readme=datapath+"/abundance_tables/J_AJ_130_2140/ReadMe").to_pandas()
    df = df[df["f_[Fe/H]"]=="b"]
    return df

def load_roediger_gcs():
    """ Processed from https://www.physics.queensu.ca/Astro/people/Stephane_Courteau/roediger2014/index.html """
    df = ascii.read(datapath+"/globular_clusters/roediger_tab.csv").to_pandas()
    return df

def load_venn04():
    # tab = Table.read(datapath+"/abundance_tables/venn04.fits")
    #for col in ["Name","n_Name","SimbadName"]:
    #    tab[col] = tab[col].astype(str)
    #tab.rename_column("__Fe_H_","[Fe/H]")
    #for elem in ["Mg","Ca","Ti","Na","Ni","Y","Ba","La","Eu"]:
    #    tab.rename_column("__{}_Fe_".format(elem), "[{}/Fe]".format(elem))
    #    tab["ul"+elem.lower()] = False
    tab = Table.read(datapath+"/abundance_tables/venn04.txt", format="ascii")
    tab.rename_columns(["Tn","Tk","Ha"],["Thin","Thick","Halo"])
    tab["Dwarf"] = (tab["Halo"] + tab["Thin"] + tab["Thick"] == 0).astype(float)
    for elem in ["Mg","Ca","Ti","Na","Ni","Y","Ba","La","Eu"]:
        tab[ulcol(elem)] = False
    tab.rename_column("[a/Fe]", "_a_Fe_")
    df = tab.to_pandas()
    XH_from_XFe(df)
    eps_from_XH(df)
    return df
def load_venn04_halo():
    df = load_venn04()
    return df[df["Halo"] > 0.5]
def load_venn04_thin():
    df = load_venn04()
    return df[df["Thin"] > 0.5]
def load_venn04_thick():
    df = load_venn04()
    return df[df["Thick"] > 0.5]
def load_venn04_dsph():
    df = load_venn04()
    return df[(df["Halo"] + df["Thin"] + df["Thick"]) == 0]
def load_venn04_mw():
    df = load_venn04()
    return df[(df["Halo"] + df["Thin"] + df["Thick"]) > 0]

def load_hill19_sculptor():
    tab = Table.read(datapath+"/abundance_tables/hill19_sculptor.fits")
    tab["Star"] = tab["Star"].astype(str)
    tab.rename_column("Star","Name")
    tab.rename_column("__Fe_H_", "[Fe/H]")
    tab.rename_column("e__Fe_H_", "e_fe")
    tab["ulfe"] = False
    for elem in ["O","Na","Mg","Si","Ca","Sc","Cr","Co","Ni","Zn","Ba","La","Nd","Eu"]:
        tab.rename_column("__{}_Fe_".format(elem), "[{}/Fe]".format(elem))
        tab.rename_column("e__{}_Fe_".format(elem), "e_{}".format(elem.lower()))
        tab["ul"+elem.lower()] = False
    tab.rename_column("__TiII_Fe_", "[Ti/Fe]")
    tab.rename_column("e__TiII_Fe_", "e_ti")
    tab["ulti"] = False
    tab["ulfe"] = False
    
    df = tab.to_pandas()
    XH_from_XFe(df)
    eps_from_XH(df)
    
    df.rename(columns={"__TiI_Fe_":"[Ti I/Fe]",
                       "e__TiI_Fe_":"e_ti1",
                       "__FeII_Fe_":"[Fe II/Fe]",
                       "e__FeII_Fe_":"e_fe2"}, inplace=True)
    df["galaxy"] = "Scl"
    df["Loc"] = "DW"
    df["Reference"] = "HIL19"
    return df
def load_letarte10_fornax():
    tab = Table.read(datapath+"/abundance_tables/letarte10_fornax.fits")
    tab["Star"] = tab["Star"].astype(str)
    tab.rename_column("Star","Name")
    for col in tab.colnames:
        if col.startswith("o__"): tab.remove_column(col)
    elemmap = {"NaI":"Na", "MgI":"Mg", "SiI":"Si", "CaI":"Ca", "TiII":"Ti",
               "CrI":"Cr", "NiI":"Ni", "YII":"Y",
               "BaII":"Ba","LaII":"La","NdII":"Nd","EuII":"Eu"}
    #"FeI":"Fe"
    for e1, e2 in elemmap.items():
        tab.rename_column("__{}_Fe_".format(e1), "[{}/Fe]".format(e2))
        tab.rename_column("e__{}_Fe_".format(e1), "e_{}".format(e2.lower()))
        tab[ulcol(e2)] = False
    tab["ulfe"] = False
    tab.rename_column("__FeI_H_", "[Fe/H]")
    tab.rename_column("e__FeI_H_", "e_fe")
    df = tab.to_pandas()
    XH_from_XFe(df)
    eps_from_XH(df)
    df.rename(columns={"__FeII_H_":"[Fe II/H]",
                       "e__FeII_H_":"e_fe2",
                       "__TiI_Fe_":"[Ti I/Fe]",
                       "e__TiI_Fe_":"e_ti1"},
              inplace=True)
    df["galaxy"] = "Fnx"
    df["Loc"] = "DW"
    df["Reference"] = "LET10"
    return df
def load_lemasle12_carina():
    tab = Table.read(datapath+"/abundance_tables/lemasle12_carina.fits")
    tab["Name"] = tab["Name"].astype(str)
    #tab.rename_column("Star","Name")
    for col in tab.colnames:
        if col.startswith("o__"): tab.remove_column(col)
    elemmap = {"Na1":"Na", "Mg1":"Mg", "Si1":"Si", "Ca1":"Ca", "Ti2":"Ti",
               "Sc2":"Sc", "Cr1":"Cr", "Co1":"Co", "Ni1":"Ni", "Y2":"Y",
               "Ba2":"Ba","La2":"La","Nd2":"Nd","Eu2":"Eu"}
    #"FeI":"Fe"
    for e1, e2 in elemmap.items():
        tab.rename_column("__{}_H_".format(e1), "[{}/H]".format(e2))
        tab.rename_column("e__{}_H_".format(e1), "e_{}".format(e2.lower()))
        tab[ulcol(e2)] = False
    tab["ulfe"] = False
    tab.rename_column("__Fe1_H_", "[Fe/H]")
    tab.rename_column("e__Fe1_H_", "e_fe")
    df = tab.to_pandas()
    XFe_from_XH(df)
    eps_from_XH(df)
    df.rename(columns={"__Fe2_H_":"[Fe II/H]",
                       "e__Fe2_H_":"e_fe2",
                       "__Ti1_H_":"[Ti I/H]",
                       "e__Ti1_H_":"e_ti1"},
              inplace=True)
    df["galaxy"] = "Car"
    df["Loc"] = "DW"
    df["Reference"] = "LEM12"
    return df

def load_battaglia17():
    df = Table.read(datapath+"/abundance_tables/battaglia17.txt", format="ascii.fixed_width_two_line").to_pandas()
    XH_from_XFe(df)
    eps_from_XH(df)
    for col in epscolnames(df):
        ul = ulcol(col)
        if ul in df:
            df[ul] = df[ul] == 1
        else:
            df[ul] = False
    return df

def load_apogee_sgr():
    """
    APOGEE_DR16 
    
    STARFLAG == 0, ASPCAPFLAG == 0, VERR < 0.2, SNR > 70
    TEFF > 3700, LOGG < 3.5
    (142775 STARS)
    
    Within 1.5*342.7 arcmin of (RA, Dec) = (283.747, -30.4606)
    (2601 STARS)

    100 < VHELIO_AVG < 180
    -3.2 < GAIA_PMRA < -2.25
    -1.9 < GAIA_PMDEC < -0.9
    (400 STARS)
    """
    tab = Table.read(datapath+"/abundance_tables/apogee_sgr.fits")
    tab.rename_column("APOGEE_ID","Name")
    cols_to_keep = ["Name","RA","DEC","M_H","M_H_ERR","ALPHA_M","ALPHA_M_ERR","TEFF","TEFF_ERR","LOGG","LOGG_ERR",
                    "VMICRO",]
    tab.rename_column("FE_H","[Fe/H]"); cols_to_keep.append("[Fe/H]")
    tab.rename_column("FE_H_ERR","e_fe"); cols_to_keep.append("e_fe")
    tab["ulfe"] = False; cols_to_keep.append("ulfe")
    for el in ["C","N","O","NA","MG","AL","SI","P","S","K","CA","TI","V","CR","MN","CO","NI","CU","CE"]:
        elem = getelem(el)
        tab["{}_FE_ERR".format(el)][tab["{}_FE".format(el)] < -9000] = np.nan
        tab["{}_FE".format(el)][tab["{}_FE".format(el)] < -9000] = np.nan
        tab.rename_column("{}_FE".format(el),"[{}/Fe]".format(elem))
        tab.rename_column("{}_FE_ERR".format(el),"e_{}".format(elem.lower()))
        tab[ulcol(elem)] = False
        cols_to_keep.extend(["[{}/Fe]".format(elem),"e_{}".format(elem.lower()),ulcol(elem)])
    df = tab[cols_to_keep].to_pandas()
    XH_from_XFe(df)
    eps_from_XH(df)
    df["galaxy"] = "Sgr"
    df["Loc"] = "DW"
    df["Reference"] = "APOGEE_DR16"
    return df

