# HeteroSV

## Overview

HeteroSV is a self-supervised caller of structural variants (SVs) in long-read metagenome series. It co-assembles the samples into one graph, gives every node an initial call with simple motif rules, and trains a heterophily-aware graph neural network on these calls, on the same graph, to call every node. It needs no reference genome, no ground truth, and no thresholds tuned to the dataset. This repository contains the method, a small example, and the scripts that reproduce the results in the main body of the paper.

<p align="center"><img src="docs/overview.png" alt="Overview of HeteroSV" width="100%"></p>

```
heterosv/         the method
run_heterosv.py   from reads to SV calls
example/          toy reads for the Quick Start
experiments/      one script per table or figure of the paper; prepare/ to rebuild the data
data/             download scripts (the data themselves are not in the repository)
paper_results/    outputs of the paper's own run, for comparison
envs/             conda environments of the tools used only to rebuild the data
third_party/      Rhea's code
```

## Installation

Linux with [conda](https://docs.conda.io/). A GPU is optional: on the toy example, HeteroSV's calling step takes seconds on a CPU.

```bash
git clone https://github.com/aliaaz99/HeteroSV.git
cd HeteroSV
conda env create -f environment.yml
conda activate heterosv
```

`environment.yml` installs metaFlye, minigraph, minimap2, seqkit, and sra-tools, and then the Python packages of `requirements.txt` with pip. If you already have these tools, `pip install -r requirements.txt` is enough. The default PyTorch wheel on Linux uses CUDA 11.7; see [PyTorch's previous versions](https://pytorch.org/get-started/previous-versions/#v1131) for other builds. Drawing the figures also needs LaTeX and dvipng.

## Quick Start

The example is the toy dataset released with Rhea: reads of two *E. coli* variants at two time points.

**1. Get the reads.**

```bash
bash example/download.sh
```

**2. Call SVs.** co-assembly, alignment of each sample to the graph, coverage, initial calls, and ten trained networks whose majority call is the result.

```bash
python run_heterosv.py example/t0.fasta example/t1.fasta --out heterosv_out
```

**3. Read the calls.** The run prints the number of SV nodes per type: 12 to 17 of 32 to 36 nodes in our runs, because metaFlye's graph differs slightly between runs.

```bash
head heterosv_out/step0/sv_calls.tsv
```

```
node_id  sv_type       node_length  cov_t0  cov_t1  lfc     seed_agreement
1        complex gain  1134         0.0     5.1921  2.3763  1.0
6        complex gain  827          0.0     2.9938  1.582   0.9
```

`sv_calls.tsv` lists the graph nodes called as an SV, `sv_nodes.fasta` their sequences, and `all_nodes.tsv` every node. One folder `step<k>/` is written per pair of consecutive samples. Calls are per node: the two branches of a complex indel, for example, are two rows.

For your own data, give the reads in series order and name their type as metaFlye does (`--read-type nano-raw`, `nano-hq`, `pacbio-hifi`, ...). `python run_heterosv.py --help` lists the options. The defaults are the configuration of the paper's methods, `--tp 0.5 --ts 0.5`.

## Reproducing Paper Results

The data behind the paper are on Zenodo ([doi:10.5281/zenodo.23818810](https://doi.org/10.5281/zenodo.23818810)): the co-assembly graphs, read alignments, ground truth, reads, and outputs of the other callers. Reproduce the results from these data, as in Option 1. Assemblers and simulators do not give the same output twice, so data regenerated from scratch (Option 2) are a new version of the datasets and give different numbers.

Each script of `experiments/` reproduces one result of the main body and writes it to `results/<name>/`. Run them from the root of the repository, in the `heterosv` environment.

| Script | Reproduces | Output in `results/` | Time, retraining |
|---|---|---|---|
| `experiments/00_fig1_overview.sh` | Figure 1 | `fig1_overview/fig1.pdf` | seconds |
| `experiments/01_table1_benchmark.sh` | Table 1 | `table1_benchmark/table1.md` | 25 min |
| `experiments/02_fig2_heterophily.sh` | Figure 2 | `fig2_heterophily/fig2.pdf` | 75 min |
| `experiments/03_fig3_learning_step.sh` | Figure 3 | `fig3_learning_step/fig3.pdf` | 70 min |
| `experiments/04_fig4_table2_cheese.sh` | Figure 4, Table 2 | `fig4_table2_cheese/fig4.pdf`, `table2.md` | 1 h |
| `experiments/05_fig5_speed.sh` | Figure 5 | `fig5_speed/fig5.pdf` | seconds |

Times are for one GPU and 16 threads on a shared machine. With `USE_PAPER_CALLS=1` (below), scripts 01--03 take about a minute and script 04 takes about 40 minutes. To list the numbers stated in the text next to the paper's values:

```bash
PYTHONPATH=.:experiments/src python experiments/src/paper_numbers.py   # writes results/paper_numbers.md
```

### Option 1: from the data on Zenodo (recommended)

```bash
bash data/download.sh               # Zymo from OSF, prepared data from Zenodo: 1 GB, 3 GB unpacked
bash data/download_cheese_reads.sh  # only for script 04: cheese reads from SRA, 5.9 GB
bash experiments/01_table1_benchmark.sh
```

This repeats the training, scoring, and read-support tests. `DEVICE=cpu` and `THREADS=32` set the device and the threads.

**Exact numbers.** Training on a GPU is not bit-reproducible: with the same seed, a few nodes change between runs. Retraining therefore gives values close to the paper's (in our rerun, Table 1 differed from the paper by at most 0.004, and Table 2 gave 179 of 578 supported calls and 295 split-control calls for HeteroSV, against 179 of 571 and 293). To get the paper's numbers exactly, use the calls of its own run, stored in `paper_results/`, and repeat everything else:

```bash
USE_PAPER_CALLS=1 bash experiments/01_table1_benchmark.sh
```

### Option 2: from scratch

This rebuilds what `data/download.sh` provides, with the extra conda environments, more disk, and much more time. Afterwards, run the scripts as in Option 1.

```bash
conda env create -f envs/simulate.yml     # NanoSim, SURVIVOR
conda env create -f envs/baselines.yml    # MetaBAT 2, MUMmer, skani, Sniffles2
conda env create -f envs/annotation.yml   # geNomad, sourmash
bash experiments/prepare/simulate_series.sh    # reads, graphs, ground truth of the simulated series
bash experiments/prepare/baselines.sh          # the callers that compare two samples
bash data/download_cheese_reads.sh
bash experiments/prepare/cheese_assembly.sh    # cheese co-assembly (4.4 h, 48 GB), baselines
bash experiments/prepare/cheese_annotation.sh  # mobile elements and host species (databases of 5 GB)
```

**The numbers will differ from the paper's.** SURVIVOR draws its SVs from the clock, NanoSim's reads differ between runs even with a fixed seed, and metaFlye's graph differs between runs on the same reads. To start closer to the paper, unpack `heterosv_simulated_reads.tar.gz` from Zenodo into `data/` first, which keeps the paper's SVs and reads. For example, on three regenerated graphs of the low-depth series, HeteroSV's node macro-F1 was 0.916--0.949 (paper: 0.949) and Rhea's 0.845--0.919 (paper: 0.900).

### Notes

- **Figure 5** is drawn from the timings of the paper's run on one NVIDIA A100 GPU. `RERUN_TIMINGS=1` measures them on your machine; Rhea needs 78 GB of memory at 22,230 nodes, and its 0.9 TB at 74,100 nodes in the paper is an estimate, not a run.
- **Zymo:** the callers that compare two samples were not run on Zymo, so Table 1 has no values for them there. The paper reports the recall from Rhea's paper for the MAG workflow.
- **Cheese settings:** the cheese results use `T_S = 0`, chosen without labels with the split-sample control (`pick` in `experiments/src/cheese/loosen.py`; the paper's appendix states this). All other datasets use `T_S = 0.5`.
- **CPU:** repeated runs on one machine are identical, but they are not the paper's GPU runs.

## License and Citation

HeteroSV is released under the [MIT License](LICENSE). If you use it, please cite:

> Ali Azizpour, Todd J. Treangen, and Santiago Segarra. *HeteroSV: Heterophilic Graph Learning for Structural Variant Calling in Metagenomes.*

`third_party/rhea/` contains code of [Rhea](https://github.com/treangenlab/rhea) (MIT License, Kristen Curry and Todd J. Treangen), unchanged in `rhea.py` and with sparse edge counts in `rhea_sparse.py`. The other tools are installed from their own distributions, under their own licenses: [metaFlye](https://github.com/fenderglass/Flye), [minigraph](https://github.com/lh3/minigraph), [minimap2](https://github.com/lh3/minimap2), [NanoSim](https://github.com/bcgsc/NanoSim), [SURVIVOR](https://github.com/fritzsedlazeck/SURVIVOR), [MUM&Co](https://github.com/SAMtoBAM/MUMandCo), [MetaBAT 2](https://bitbucket.org/berkeleylab/metabat), [skani](https://github.com/bluenote-1577/skani), [Sniffles2](https://github.com/fritzsedlazeck/Sniffles), [geNomad](https://github.com/apcamargo/genomad), and [sourmash](https://github.com/sourmash-bio/sourmash).
