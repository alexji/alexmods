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
'./b11tw0.25.dat' u 4:5 ti 'B11{/Symbol b}0.25' pt  5 lc 1,\
'./b11tw1.00.dat' u 4:5 ti 'B11{/Symbol b}1.00' pt  7 lc 2,\
'./b12tw0.25.dat' u 4:5 ti 'B12{/Symbol b}0.25' pt  9 lc 3,\
'./b12tw1.00.dat' u 4:5 ti 'B12{/Symbol b}0.25' pt 11 lc 4,\
'./b12tw4.s3.dat' u 4:5 ti 'B12{/Symbol b}4.00' pt 13 lc 5


se ou 'fig_x-z.pdf'

se xr [20:100]
se yr [1e-8:1]

se xti 0, 10
se mxti 5

se xl 'proton number, Z'

p \
'./b11tw0.25.dat' u 2:5 ti 'B11{/Symbol b}0.25' pt  5 lc 1,\
'./b11tw1.00.dat' u 2:5 ti 'B11{/Symbol b}1.00' pt  7 lc 2,\
'./b12tw0.25.dat' u 2:5 ti 'B12{/Symbol b}0.25' pt  9 lc 3,\
'./b12tw1.00.dat' u 2:5 ti 'B12{/Symbol b}0.25' pt 11 lc 4,\
'./b12tw4.s3.dat' u 2:5 ti 'B12{/Symbol b}4.00' pt 13 lc 5
