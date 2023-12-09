

README Document: Nucleosynthesis yields of NTT15


This is a set of nucleosynthesis yields calculated by NTT15 (Nishimura, Takiwaki and Thielemann, 2015, ApJ 810:109), which includes resutls of five different explosion models. These yields are final abundances after decay, so that only stable nuclei (and long half-life Th and T) are listed.

Note: NTT paper denoted that the name of models as 'B11*beta*025' using Greek character *beta*, but we cannnot use this expression. Here, we used 'TW' instead of *beta*, which stands origional meaning (for more details, see, Section 2.1 of NTT15).


- notation

name: elemental symbol with mass number
Z   : proton number
N   : neutron number
A   : mass number, Z + N
X   : abundance by mass fraction, where the sum of X = 1
Y   : abundance by mole fraction, where Y = X/A


- the mass of ejecta (in Solar Mass: 1.989e33 g)

B11TW0.25:   2.68000e-02
B11TW1.00:   2.15000e-02
B12TW0.25:   3.55000e-02
B12TW1.00:   4.37000e-02
B12TW4.00:   8.57000e-02


- other tools

plot_x-az.gp: plot script for Gnuplot, we only tested on the latest version 5.0



- update log
  + 2015.10.01: first release (N. Nishimura)
