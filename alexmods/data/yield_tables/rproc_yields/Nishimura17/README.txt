

README Document: Nucleosynthesis yields of NST*2017


This is a set of nucleosynthesis yields calculated by NST*2017 (Nishimura, Sawai, Takiwaki, Yamada and Thielemann, 2017, ApJL 836:L21), which includes resutls of five different explosion models. These yields are final abundances after decay, so that only stable nuclei (and long half-life Th and T) are listed.

Note that, the name of models is written as L0.10, of which number is a scale factor of neutrino luminosity in the paper.


- notation

name: elemental symbol with mass number
Z   : proton number
N   : neutron number
A   : mass number, Z + N
X   : abundance by mass fraction, where the sum of X = 1
Y   : abundance by mole fraction, where Y = X/A


- the mass of ejecta (in Solar Mass: 1.989e33 g)

L0.10:    7.66225e-03
L0.20:    1.34505e-02
L0.30:    2.09197e-02
L0.40:    3.19371e-02
L0.50:    6.17773e-02
L0.60:    1.19465e-01
L0.75:    2.00467e-01
L1.00:    2.38585e-01
L1.25:    2.61565e-01


- other tools

plot_x-az.gp: plot script for Gnuplot, we only tested on the latest version 5.0



- update log
  + 2017.01.08: first release (N. Nishimura)
