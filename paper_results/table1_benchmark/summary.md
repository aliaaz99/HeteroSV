# HeteroSV (frozen) on the simulated series

## Main labellers (node macro-F1; per-SV recall / precision as in rhea's paper; learned = mean of seeds)

| series | labeller | macro_f1 (learned) | macro_f1 (rules alone) | sv_precision (learned) | sv_precision (rules alone) | sv_recall (learned) | sv_recall (rules alone) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_sv | flank rule 0.25/0.5 | 0.898 | 0.845 | 0.979 | 0.966 | 0.751 | 0.680 |
| long_sv | flank rule 0.5/0.5 | 0.897 | 0.841 | 0.973 | 0.971 | 0.753 | 0.670 |
| long_sv | rhea's rules | 0.882 | 0.870 | 0.953 | 0.953 | 0.757 | 0.736 |
| low_depth | flank rule 0.25/0.5 | 0.950 | 0.933 | 0.947 | 0.943 | 0.590 | 0.562 |
| low_depth | flank rule 0.5/0.5 | 0.949 | 0.916 | 0.959 | 0.947 | 0.582 | 0.538 |
| low_depth | rhea's rules | 0.935 | 0.900 | 0.909 | 0.886 | 0.589 | 0.560 |
| partial_change | flank rule 0.25/0.5 | 0.962 | 0.935 | 0.956 | 0.972 | 0.666 | 0.626 |
| partial_change | flank rule 0.5/0.5 | 0.956 | 0.883 | 0.964 | 0.979 | 0.654 | 0.564 |
| partial_change | rhea's rules | 0.892 | 0.864 | 0.946 | 0.930 | 0.602 | 0.582 |
| replica | flank rule 0.25/0.5 | 0.967 | 0.942 | 0.966 | 0.956 | 0.636 | 0.610 |
| replica | flank rule 0.5/0.5 | 0.970 | 0.935 | 0.972 | 0.967 | 0.636 | 0.594 |
| replica | rhea's rules | 0.953 | 0.923 | 0.933 | 0.898 | 0.640 | 0.632 |
| uneven_abundance | flank rule 0.25/0.5 | 0.960 | 0.940 | 0.970 | 0.960 | 0.591 | 0.570 |
| uneven_abundance | flank rule 0.5/0.5 | 0.959 | 0.933 | 0.966 | 0.968 | 0.592 | 0.552 |
| uneven_abundance | rhea's rules | 0.953 | 0.919 | 0.950 | 0.905 | 0.591 | 0.574 |
| zymo | flank rule 0.25/0.5 | 0.864 | 0.852 | 0.957 | 0.944 | 0.582 | 0.568 |
| zymo | flank rule 0.5/0.5 | 0.864 | 0.831 | 0.958 | 0.944 | 0.581 | 0.540 |
| zymo | rhea's rules | 0.892 | 0.877 | 0.929 | 0.924 | 0.670 | 0.658 |

## Gain of the learning step, 95% bootstrap interval over nodes

| series | labeller | nodes | gain | ci_low | ci_high | p_gain_le_0 |
| --- | --- | --- | --- | --- | --- | --- |
| zymo | rhea's rules | all | 0.015 | 0.002 | 0.029 | 0.008 |
| zymo | rhea's rules | uncertain | 0.061 | 0.022 | 0.108 | 0.000 |
| zymo | rhea's rules | confident | 0.001 | -0.010 | 0.012 | 0.444 |
| zymo | flank rule 0.25/0.5 | all | 0.012 | 0.004 | 0.022 | 0.004 |
| zymo | flank rule 0.5/0.5 | all | 0.032 | 0.020 | 0.048 | 0.000 |
| replica | rhea's rules | all | 0.030 | 0.014 | 0.046 | 0.000 |
| replica | rhea's rules | uncertain | 0.129 | 0.072 | 0.195 | 0.000 |
| replica | rhea's rules | confident | 0.000 | -0.009 | 0.010 | 0.464 |
| replica | flank rule 0.25/0.5 | all | 0.024 | 0.012 | 0.039 | 0.000 |
| replica | flank rule 0.5/0.5 | all | 0.034 | 0.020 | 0.050 | 0.000 |
| long_sv | rhea's rules | all | 0.012 | 0.002 | 0.023 | 0.014 |
| long_sv | rhea's rules | uncertain | 0.041 | 0.013 | 0.073 | 0.000 |
| long_sv | rhea's rules | confident | 0.001 | -0.009 | 0.011 | 0.410 |
| long_sv | flank rule 0.25/0.5 | all | 0.054 | 0.037 | 0.072 | 0.000 |
| long_sv | flank rule 0.5/0.5 | all | 0.056 | 0.039 | 0.074 | 0.000 |
| low_depth | rhea's rules | all | 0.035 | 0.015 | 0.055 | 0.000 |
| low_depth | rhea's rules | uncertain | 0.133 | 0.077 | 0.196 | 0.000 |
| low_depth | rhea's rules | confident | 0.005 | -0.013 | 0.024 | 0.326 |
| low_depth | flank rule 0.25/0.5 | all | 0.017 | 0.001 | 0.035 | 0.022 |
| low_depth | flank rule 0.5/0.5 | all | 0.033 | 0.017 | 0.051 | 0.000 |
| partial_change | rhea's rules | all | 0.028 | 0.014 | 0.045 | 0.000 |
| partial_change | rhea's rules | uncertain | 0.106 | 0.053 | 0.170 | 0.000 |
| partial_change | rhea's rules | confident | 0.002 | -0.006 | 0.011 | 0.304 |
| partial_change | flank rule 0.25/0.5 | all | 0.028 | 0.012 | 0.045 | 0.000 |
| partial_change | flank rule 0.5/0.5 | all | 0.073 | 0.051 | 0.096 | 0.000 |
| uneven_abundance | rhea's rules | all | 0.034 | 0.018 | 0.053 | 0.000 |
| uneven_abundance | rhea's rules | uncertain | 0.136 | 0.078 | 0.206 | 0.000 |
| uneven_abundance | rhea's rules | confident | 0.003 | -0.008 | 0.014 | 0.326 |
| uneven_abundance | flank rule 0.25/0.5 | all | 0.020 | 0.006 | 0.035 | 0.000 |
| uneven_abundance | flank rule 0.5/0.5 | all | 0.026 | 0.010 | 0.044 | 0.000 |

## rhea's least / most confident calls

| series | calls | macro_f1_uncertain | macro_f1_confident |
| --- | --- | --- | --- |
| long_sv | learned | 0.966 | 0.868 |
| long_sv | rules alone | 0.925 | 0.867 |
| low_depth | learned | 0.831 | 0.961 |
| low_depth | rules alone | 0.698 | 0.956 |
| partial_change | learned | 0.871 | 0.898 |
| partial_change | rules alone | 0.765 | 0.895 |
| replica | learned | 0.907 | 0.961 |
| replica | rules alone | 0.779 | 0.961 |
| uneven_abundance | learned | 0.926 | 0.958 |
| uneven_abundance | rules alone | 0.790 | 0.956 |
| zymo | learned | 0.856 | 0.901 |
| zymo | rules alone | 0.795 | 0.900 |

## Threshold sensitivity (12 flank-rule settings): macro-F1 range

| series | calls | min | max | mean |
| --- | --- | --- | --- | --- |
| long_sv | learned | 0.759 | 0.902 | 0.881 |
| long_sv | rules alone | 0.755 | 0.879 | 0.827 |
| low_depth | learned | 0.828 | 0.954 | 0.934 |
| low_depth | rules alone | 0.799 | 0.933 | 0.891 |
| partial_change | learned | 0.833 | 0.965 | 0.923 |
| partial_change | rules alone | 0.669 | 0.945 | 0.844 |
| replica | learned | 0.830 | 0.970 | 0.947 |
| replica | rules alone | 0.819 | 0.950 | 0.911 |
| uneven_abundance | learned | 0.841 | 0.961 | 0.945 |
| uneven_abundance | rules alone | 0.810 | 0.940 | 0.898 |
| zymo | learned | 0.736 | 0.872 | 0.854 |
| zymo | rules alone | 0.726 | 0.852 | 0.810 |

## Best flank-rule thresholds per series (truth) and cost of Zymo's best (0.25/0.5)

| series | labeller | macro_f1 | zymo_best_macro_f1 | transfer_loss |
| --- | --- | --- | --- | --- |
| long_sv | flank rule 0.5/0.0 | 0.879 | 0.845 | 0.034 |
| low_depth | flank rule 0.25/0.5 | 0.933 | 0.933 | 0.000 |
| partial_change | flank rule 0.0/0.5 | 0.945 | 0.935 | 0.010 |
| replica | flank rule 0.0/0.5 | 0.950 | 0.942 | 0.008 |
| uneven_abundance | flank rule 0.25/0.5 | 0.940 | 0.940 | 0.000 |
| zymo | flank rule 0.25/0.5 | 0.852 | 0.852 | 0.000 |
