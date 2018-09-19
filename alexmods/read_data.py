from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from astropy import table
from astropy import coordinates as coord
from astropy import units as u

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
    return 'e_'+getelem(elem,lower=True)
def ulcol(elem):
    return 'ul'+getelem(elem,lower=True)
def XHcol(elem,keep_species=False):
    return '['+getelem(elem,keep_species=keep_species)+'/H]'
def XFecol(elem,keep_species=False):
    return '['+getelem(elem,keep_species=keep_species)+'/Fe]'
def ABcol(elems):
    """ Note: by default the data does not have [A/B] """
    A,B = elems
    return '['+getelem(A)+'/'+getelem(B)+']'

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
    epscols = epscolnames(df)
    asplund = get_solar(epscols)
    for col in epscols:
        if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
        df[XHcol(col)] = df[col] - float(asplund[col])
def XFe_from_eps(df):
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
    XHcols = XHcolnames(df)
    asplund = get_solar(XHcols)
    for col in XHcols:
        if epscol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(epscol(col)))
        df[epscol(col)] = df[col] + float(asplund[col])
def XFe_from_XH(df):
    XHcols = XHcolnames(df)
    assert '[Fe/H]' in XHcols
    feh = df['[Fe/H]']
    for col in XHcols:
        if col=='[Fe/H]': continue
        if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
        df[XFecol(col)] = df[col] - feh
def eps_from_XFe(df):
    XFecols = XFecolnames(df)
    assert '[Fe/H]' in df
    asplund = get_solar(XFecols)
    feh = df['[Fe/H]']
    for col in XFecols:
        df[epscol(col)] = df[col] + feh + float(asplund[col])
def XH_from_XFe(df):
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
    ## the units of this table are mol/g?
    ## I want Msun/amu?
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
