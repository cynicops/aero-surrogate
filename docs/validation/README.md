# Validation data

`reference_lift.csv` is read by `aerosurrogate phase0` (Gate 0). It does not exist yet:
**no published values have been entered**, because the sources could not be opened in
session 1. Never type values from memory; copy them from the paper/thesis and cite the
table or figure (digitised figures must say so).

Format (comment lines start with `#`):

```
# source: Zerihan & Zhang (2000) J. Aircraft 37(6), Table N / Fig. N (digitised: yes/no)
# section: <coordinates file>, incidence <deg>, Re <value>, sectional or 3D: <...>
h_over_c,cl_downforce
0.448,<value>
...
```

`cl_downforce` is positive for downforce (= -cl in our convention). If the paper's forces are
3D (finite span with endplates), they are not directly comparable with 2D CFD; compare
sectional values (integrated centre-span pressures) instead and say so.
