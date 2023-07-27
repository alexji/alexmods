se te pdf

se ou 'fig_x-a.pdf'

se log y

se xr [60:240]
se yr [1e-8:1]

se xl 'mass number, A'
se yl 'mass fraction, X'

se xti 0, 50
se mxti 5

se fo y '10^{%L}'
se k sp 1.1

p \
'./L0.20.dat' u 4:5 ti 'm-model' pt  7 lc 2,\
'./L0.60.dat' u 4:5 ti 'i-model' pt  9 lc 3,\
'./L1.00.dat' u 4:5 ti 'h-model' pt 11 lc 4


se ou 'fig_x-z.pdf'

se xr [20:100]
se yr [1e-8:1]

se xti 0, 10
se mxti 5

se xl 'proton number, Z'

p \
'./L0.20.dat' u 2:5 ti 'm-model' pt  7 lc 2,\
'./L0.60.dat' u 2:5 ti 'i-model' pt  9 lc 3,\
'./L1.00.dat' u 2:5 ti 'h-model' pt 11 lc 4
