# Table 1: SV calling on the six simulated datasets

Learned methods: mean of 10 seeds. Flank best: the best of 12 threshold settings for each dataset, chosen with the ground truth. The callers that compare the two samples were not run on Zymo.

## Node macro-F1

| Dataset | Rhea | Rhea + learn. | Flank rule | Flank best | HeteroSV |
|---|---|---|---|---|---|
| Zymo | 0.877 | 0.892 | 0.831 | 0.852 | 0.864 |
| paper | 0.877 | 0.892 | 0.831 | 0.852 | 0.864 |
| Replica | 0.923 | 0.953 | 0.935 | 0.950 | 0.970 |
| paper | 0.923 | 0.953 | 0.935 | 0.950 | 0.970 |
| Long SVs | 0.870 | 0.882 | 0.841 | 0.879 | 0.897 |
| paper | 0.870 | 0.882 | 0.841 | 0.879 | 0.897 |
| Low depth | 0.900 | 0.935 | 0.916 | 0.933 | 0.949 |
| paper | 0.900 | 0.935 | 0.916 | 0.933 | 0.949 |
| Partial change | 0.864 | 0.892 | 0.883 | 0.945 | 0.956 |
| paper | 0.864 | 0.892 | 0.883 | 0.945 | 0.956 |
| Uneven ab. | 0.919 | 0.953 | 0.933 | 0.940 | 0.959 |
| paper | 0.919 | 0.953 | 0.933 | 0.940 | 0.959 |

## Per-SV F1

| Dataset | MAG workfl. | Assembly comp. | Sniffles2 | Rhea | Rhea + learn. | Flank rule | HeteroSV |
|---|---|---|---|---|---|---|---|
| Zymo | -- | -- | -- | 0.769 | 0.779 | 0.687 | 0.723 |
| paper | -- | -- | -- | 0.769 | 0.779 | 0.687 | 0.723 |
| Replica | 0.478 | 0.552 | 0.684 | 0.742 | 0.759 | 0.736 | 0.769 |
| paper | 0.478 | 0.552 | 0.684 | 0.742 | 0.759 | 0.736 | 0.769 |
| Long SVs | 0.473 | 0.465 | 0.597 | 0.831 | 0.844 | 0.793 | 0.849 |
| paper | 0.473 | 0.465 | 0.597 | 0.831 | 0.844 | 0.793 | 0.849 |
| Low depth | 0.487 | 0.462 | 0.429 | 0.686 | 0.715 | 0.686 | 0.724 |
| paper | 0.487 | 0.462 | 0.429 | 0.686 | 0.715 | 0.686 | 0.724 |
| Partial change | 0.084 | 0.098 | 0.031 | 0.716 | 0.736 | 0.716 | 0.779 |
| paper | 0.084 | 0.098 | 0.031 | 0.716 | 0.736 | 0.716 | 0.779 |
| Uneven ab. | 0.543 | 0.541 | 0.671 | 0.703 | 0.729 | 0.703 | 0.734 |
| paper | 0.543 | 0.541 | 0.671 | 0.703 | 0.729 | 0.703 | 0.734 |
