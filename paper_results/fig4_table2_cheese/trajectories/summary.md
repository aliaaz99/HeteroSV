# Cheese: SV allele trajectories and their hosts

Supported HeteroSV calls: 179; with an allele share in >= 3 weeks: 127; shifts (share range >= 0.3): 97; to near fixation (from 0.25-0.75 to <= 0.15 or >= 0.85): 51; with a named host: 176; within-species turnover (sweep, host < 2-fold): 43.

By host species:

| species | calls | shifts | to_fixation | turnover |
| --- | --- | --- | --- | --- |
| s__Psychrobacter sp014861395 | 52 | 47 | 44 | 27 |
| s__Serratia proteamaculans | 36 | 12 | 0 | 1 |
| s__Vreelandella citrativorans | 33 | 20 | 2 | 13 |
| s__Mesonia profundi | 30 | 14 | 4 | 0 |
| s__Alcaligenes pakistanensis | 18 | 2 | 0 | 2 |
| s__Lactococcus lactis | 3 | 0 | 0 | 0 |
| s__Pseudoalteromonas prydzensis | 1 | 0 | 0 | 0 |
| s__Vibrio litoralis | 1 | 1 | 1 | 0 |
| s__Vreelandella casei | 1 | 1 | 0 | 0 |
| s__Vreelandella sp031010195 | 1 | 0 | 0 | 0 |

Sweeps by mobile-element status:

| mge | calls | sweeps |
| --- | --- | --- |
| none | 146 | 79 |
| plasmid | 29 | 17 |
| provirus | 1 | 0 |
| virus | 3 | 1 |

Turnover cases:

| step | node_id | sv_type | species | mge | share_w2 | share_w3 | share_w4 | share_w9 | share_w13 | host_log2fc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2-3 | 153 | complex loss | s__Psychrobacter sp014861395 | plasmid | 0.380 | 0.110 | 0.110 | 0.050 | 0.070 | 0.520 |
| 2-3 | 154 | complex gain | s__Psychrobacter sp014861395 | plasmid | 0.620 | 0.890 | 0.890 | 0.950 | 0.930 | -0.520 |
| 2-3 | 157 | complex loss | s__Psychrobacter sp014861395 | plasmid | 0.440 | 0.140 | 0.110 | 0.100 | 0.090 | 0.580 |
| 2-3 | 158 | complex gain | s__Psychrobacter sp014861395 | plasmid | 0.560 | 0.860 | 0.890 | 0.900 | 0.910 | -0.580 |
| 2-3 | 451 | deletion | s__Psychrobacter sp014861395 | none | 0.560 | 0.270 | 0.140 | 0.150 | 0.110 | 0.500 |
| 2-3 | 647 | insertion | s__Vreelandella citrativorans | none | 0.200 | 0.600 | 0.270 | 0.150 | 0.170 | -0.010 |
| 2-3 | 796 | insertion | s__Alcaligenes pakistanensis | none | 0.100 | 0.570 | 0.150 | 0.160 | 0.160 | 0.000 |
| 2-3 | 806 | complex gain | s__Vreelandella citrativorans | none | 0.250 | 0.610 | 0.210 | 0.350 | 0.300 | -0.940 |
| 2-3 | 807 | complex loss | s__Vreelandella citrativorans | none | 0.750 | 0.390 | 0.790 | 0.650 | 0.700 | 0.940 |
| 2-3 | 940 | complex loss | s__Psychrobacter sp014861395 | none | 0.570 | 0.370 | 0.200 | 0.140 | 0.100 | 0.580 |
| 2-3 | 941 | deletion | s__Psychrobacter sp014861395 | none | 0.650 | 0.220 | 0.140 | 0.090 | 0.130 | 0.520 |
| 2-3 | 1069 | complex loss | s__Psychrobacter sp014861395 | plasmid | 0.460 | 0.200 | 0.160 | 0.130 | 0.080 | 0.580 |
| 2-3 | 1070 | complex gain | s__Psychrobacter sp014861395 | none | 0.540 | 0.800 | 0.840 | 0.870 | 0.920 | -0.580 |
| 2-3 | 1168 | complex loss | s__Psychrobacter sp014861395 | plasmid | 0.490 | 0.230 | 0.180 | 0.110 | 0.110 | 0.520 |
| 2-3 | 1173 | complex gain | s__Psychrobacter sp014861395 | none | 0.500 | 0.830 | 0.810 | 0.940 | 0.860 | 0.090 |
| 2-3 | 1174 | complex loss | s__Psychrobacter sp014861395 | none | 0.500 | 0.170 | 0.190 | 0.060 | 0.140 | -0.090 |
| 2-3 | 1175 | complex loss | s__Psychrobacter sp014861395 | none | 0.470 | 0.210 | 0.170 | 0.130 | 0.160 | -0.090 |
| 2-3 | 1182 | complex gain | s__Psychrobacter sp014861395 | none | 0.520 | 0.770 | 0.810 | 0.820 | 0.860 | -0.580 |
| 2-3 | 1183 | complex loss | s__Psychrobacter sp014861395 | none | 0.480 | 0.230 | 0.190 | 0.180 | 0.140 | 0.580 |
| 2-3 | 1196 | deletion | s__Psychrobacter sp014861395 | none | 0.590 | 0.350 | 0.150 | 0.150 | 0.100 | 0.580 |
| 2-3 | 3084 | complex gain | s__Psychrobacter sp014861395 | none | 0.530 | 0.790 | 0.830 | 0.870 | 0.840 | 0.090 |
| 2-3 | 3308 | complex gain | s__Psychrobacter sp014861395 | none | 0.430 | 0.630 | 0.800 | 0.860 | 0.900 | -0.580 |
| 2-3 | 3427 | complex gain | s__Psychrobacter sp014861395 | plasmid | 0.510 | 0.770 | 0.820 | 0.890 | 0.890 | -0.520 |
| 2-3 | 3437 | insertion | s__Psychrobacter sp014861395 | plasmid | 0.450 | 0.870 | 0.940 | 0.940 | 0.960 | -0.580 |
| 3-4 | 276 | insertion | s__Vreelandella citrativorans | none | 0.770 | 0.370 | 0.780 | 0.660 | 0.720 | -0.770 |
| 3-4 | 296 | deletion | s__Vreelandella citrativorans | none | 0.330 | 0.520 | 0.310 | 0.170 | 0.140 | -0.020 |
| 3-4 | 610 | complex gain | s__Psychrobacter sp014861395 | plasmid | 0.600 | 0.620 | 0.850 | 0.910 | 0.870 | -0.520 |
| 3-4 | 611 | complex loss | s__Psychrobacter sp014861395 | plasmid | 0.400 | 0.380 | 0.150 | 0.090 | 0.130 | 0.520 |
| 3-4 | 647 | deletion | s__Vreelandella citrativorans | none | 0.200 | 0.600 | 0.270 | 0.150 | 0.170 | -0.010 |
| 3-4 | 796 | deletion | s__Alcaligenes pakistanensis | none | 0.100 | 0.570 | 0.150 | 0.160 | 0.160 | 0.000 |
| 3-4 | 806 | complex loss | s__Vreelandella citrativorans | none | 0.250 | 0.610 | 0.210 | 0.350 | 0.300 | -0.940 |
| 3-4 | 807 | complex gain | s__Vreelandella citrativorans | none | 0.750 | 0.390 | 0.790 | 0.650 | 0.700 | 0.940 |
| 3-4 | 817 | complex loss | s__Vreelandella citrativorans | none | 0.370 | 0.590 | 0.320 | 0.340 | 0.280 | -0.660 |
| 3-4 | 3188 | complex loss | s__Vreelandella citrativorans | none | 0.210 | 0.470 | 0.260 | 0.160 | 0.140 | -0.020 |
| 3-4 | 3243 | complex gain | s__Vreelandella citrativorans | plasmid | 0.630 | 0.410 | 0.680 | 0.660 | 0.720 | 0.660 |
| 3-4 | 3296 | complex gain | s__Psychrobacter sp014861395 | none | 0.540 | 0.700 | 0.890 | 0.860 | 0.900 | -0.350 |
| 3-4 | 3429 | deletion | s__Psychrobacter sp014861395 | plasmid | 0.580 | 0.480 | 0.190 | 0.190 | 0.170 | 0.580 |
| 3-4 | 3430 | complex gain | s__Psychrobacter sp014861395 | none | 0.470 | 0.640 | 0.870 | 0.940 | 0.900 | -0.290 |
| 3-4 | 3431 | complex loss | s__Psychrobacter sp014861395 | none | 0.530 | 0.360 | 0.130 | 0.060 | 0.100 | 0.290 |
| 4-9 | 296 | deletion | s__Vreelandella citrativorans | none | 0.330 | 0.520 | 0.310 | 0.170 | 0.140 | -0.020 |
| 4-9 | 647 | deletion | s__Vreelandella citrativorans | none | 0.200 | 0.600 | 0.270 | 0.150 | 0.170 | -0.010 |
| 9-13 | 1172 | complex loss | s__Psychrobacter sp014861395 | none | 0.370 | 0.190 | 0.120 | 0.090 | 0.020 | 0.030 |
| 9-13 | 2676 | complex loss | s__Serratia proteamaculans | none | 0.920 | 0.800 | nan | 0.760 | 0.600 | -0.200 |
