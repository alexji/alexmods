from astropy.table import Table, vstack
import numpy as np
import os, glob
def parse_tbl(fname):
    ngc = int(os.path.basename(fname).split(".tbl")[0][3:])
    #print(ngc)
    all_data = []
    colnames = ["NGC","Ref","Star","Source","Age","AgeErr",
                "[Fe/H]","e_fe","[Mg/Fe]","e_mg","[C/Fe]","e_c","[N/Fe]","e_n","[Ca/Fe]","e_ca","[O/Fe]","e_o",
                "[Na/Fe]","e_na","[Si/Fe]","e_si","[Cr/Fe]","e_cr","[Ti/Fe]","e_ti","Comments"]
    with open(fname,"r") as fp:
        # Drop header
        line = fp.readline().strip()
        line = fp.readline().strip()
        while True:
            line = fp.readline().strip()
            if "=====" in line: break
            if ":" in line:
                ref = line.split(":")[0].strip()
                line = fp.readline() # skip blank spot
                while True:
                    line = fp.readline().strip()
                    if line == "": break
                    if "=====" in line: break
                    s = [x.strip() for x in line.split()][0:25]
                    if len(s) != 25:
                        raise RuntimeError
                    #print(len(s), line)
                    all_data.append([ngc, ref] + s)
            if "=====" in line: break
    tab = Table(rows=all_data, names=colnames,
                masked=True)
    for col in tab.colnames[4:-1]:
        tab[col].fill_value = np.nan
        ii = tab[col] == "-"
        mask = np.zeros(len(tab), dtype=bool)
        mask[ii] = True
        tab[col][ii] = np.ma.masked
        try:
            tab[col] = tab[col].astype(float)
        except Exception as e:
            tab[col] = tab[col].astype("<U3")
            tab[col].fill_value = np.nan
            tab[col] = tab[col].astype(float)
    return tab
if __name__=="__main__":
    fnames = glob.glob("NGC*.tbl")
    alltab = []
    for fname in fnames:
        tab = parse_tbl(fname)
        alltab.append(tab)
    alltab = vstack(alltab)
    alltab.write("../roediger_tab.csv",format="ascii.csv")
    
