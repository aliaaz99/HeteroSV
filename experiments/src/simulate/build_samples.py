"""Simulate the two samples (T0, T1) of every series.

Abundance profiles (NanoSim metagenome abundances are percentages of BASES per strain; 'X' is
the original strain of species X and 'X_variant' the strain carrying the SVs of the series' sv
set). rhea's inputs, recovered from the bases of its stored reads, are the D6300 composition:
12% per bacterium and 2% per yeast, constant over time per species.
  rhea_d6300   T0: original strains; T1: variant only for half of the species, 50/50 original
               and variant for the other half (common.variant_only_species / mixed_species);
               replica, short_sv, long_sv, low_depth
  partial      species totals as rhea's; variant fraction 0.2 at T0 and 0.6 at T1, every species
  uneven       bacterial species totals geometric over fold_range (random order, order_seed),
               summing to rhea's bacterial total (96%), yeasts as rhea's; constant over time;
               T1 composition as rhea_d6300
Reads: NanoSim 3.1.0 metagenome mode (with the line removed that writes every aligned read with an
empty name, patched_simulator()) with the pre-trained Zymo Even model (ERR3152364), one run
per sample with the strains present in it, seed = series seed * 10 + sample index. Aligned and
unaligned (noise) reads are merged into one fasta per sample, as in rhea's data.

  python experiments/src/simulate/build_samples.py --prepare
  python experiments/src/simulate/build_samples.py --run SERIES SAMPLE
  python experiments/src/simulate/build_samples.py --summarise
Writes data/simulated/<series>/samples/{<sample>_abundance.tsv, <sample>_genome_list.tsv,
<sample>_dna_type_list.tsv, profile.tsv}, series/<series>/reads/<sample>.nanosim.fasta and
series/<series>/reads/reads_summary.tsv.
"""
import argparse
import collections
import glob
import json
import os
import random
import shutil
import subprocess
import sys

from common import SIMULATED, SRC

ENV = os.path.dirname(shutil.which("simulator.py") or "")  # the conda environment envs/simulate.yml
SAMPLES = ("t0", "t1")
YEASTS = ("Saccharomyces_cerevisiae", "Cryptococcus_neoformans")


def patched_simulator():
    """NanoSim 3.1.0's simulator.py writes every aligned read with an empty name: its metagenome mode
    resets the name before writing it (NanoSim 3.2.3 removed that line). A copy without the line; its
    other modules are imported from the environment."""
    lines = open(os.path.join(ENV, "simulator.py")).readlines()
    resets = [k for k, l in enumerate(lines) if l.strip() == 'new_read_name = ""']
    if len(resets) != 3:
        raise ValueError("expected NanoSim 3.1.0's simulator.py")
    del lines[resets[1]]
    out = os.path.join(SIMULATED, "_inputs", "simulator_patched.py")
    with open(out, "w") as handle:
        handle.writelines(lines)
    return out


def load():
    cfg = json.load(open(os.path.join(SRC, "simulate", "series.json")))
    genomes = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(SIMULATED, "_inputs",
                                                                          "genome_list.tsv")))
    dna = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in
           open(os.path.join(SIMULATED, "_inputs", "dna_type_list.tsv"))}
    shares = {}
    with open(os.path.join(SIMULATED, "_inputs", "rhea_zymo_profile", "read_shares.tsv")) as handle:
        next(handle)
        for line in handle:
            f = line.rstrip("\n").split("\t")
            shares[f[0]] = (float(f[2]), float(f[4]))
    return cfg, genomes, dna, shares


def profiles(series, cfg, genomes, shares):
    """{sample: {strain: percent of bases}} for one series."""
    ab = cfg["series"][series]["abundance"]
    species = sorted(genomes)
    total = {sp: (2.0 if sp in YEASTS else 12.0) for sp in species}  # rhea's D6300 inputs
    variant_only = set(cfg["common"]["variant_only_species"])
    t1_variant = {sp: 1.0 if sp in variant_only else 0.5 for sp in species}
    t0_variant = {sp: 0.0 for sp in species}
    if ab != "rhea_d6300":
        if ab["type"] == "partial":
            t0_variant = {sp: ab["variant_fraction"][0] for sp in species}
            t1_variant = {sp: ab["variant_fraction"][1] for sp in species}
        elif ab["type"] == "uneven":
            bact = [sp for sp in species if sp not in YEASTS]
            ranks = list(range(len(bact)))
            random.Random(ab["order_seed"]).shuffle(ranks)
            w = {sp: ab["fold_range"] ** (r / (len(bact) - 1)) for sp, r in zip(bact, ranks)}
            bact_total = sum(total[sp] for sp in bact)
            total.update({sp: bact_total * w[sp] / sum(w.values()) for sp in bact})
        else:
            raise ValueError(ab)
    prof = {}
    for s, var in (("t0", t0_variant), ("t1", t1_variant)):
        p = {}
        for sp in species:
            if var[sp] < 1:
                p[sp] = total[sp] * (1 - var[sp])
            if var[sp] > 0:
                p[sp + "_variant"] = total[sp] * var[sp]
        tot = sum(p.values())
        prof[s] = {k: 100 * v / tot for k, v in p.items()}
    return prof


def prepare(cfg, genomes, dna, shares, only=None):
    for series, sc in cfg["series"].items():
        if only and series not in only:
            continue
        base = os.path.join(SIMULATED, series, "samples")
        os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(SIMULATED, series, "reads"), exist_ok=True)
        variants = os.path.join(SIMULATED, "_svsets", sc["sv_set"], "variants")
        prof = profiles(series, cfg, genomes, shares)
        rows = []
        for s in SAMPLES:
            strains = sorted(prof[s])
            vals = [round(prof[s][st], 6) for st in strains]
            vals[-1] = round(100 - sum(vals[:-1]), 6)  # sum exactly 100
            with open(os.path.join(base, s + "_abundance.tsv"), "w") as handle:
                handle.write("Size\t{}\n".format(sc["reads_per_sample"]))
                handle.writelines("{}\t{}\n".format(st, v) for st, v in zip(strains, vals))
            with open(os.path.join(base, s + "_genome_list.tsv"), "w") as gl, \
                    open(os.path.join(base, s + "_dna_type_list.tsv"), "w") as dl:
                for st in strains:
                    sp = st.replace("_variant", "")
                    if st.endswith("_variant"):
                        path, chrom = os.path.join(variants, sp + ".fasta"), dna[sp][1] + "-mutated"
                    else:
                        path, chrom = genomes[sp], dna[sp][1]
                    gl.write("{}\t{}\n".format(st, path))
                    dl.write("{}\t{}\t{}\n".format(st, chrom, dna[sp][2]))
            for st, v in zip(strains, vals):
                rows.append((s, st, v))
        with open(os.path.join(base, "profile.tsv"), "w") as handle:
            handle.write("sample\tstrain\tpercent_bases\n")
            handle.writelines("{}\t{}\t{:.4f}\n".format(*r) for r in rows)
        print(series, "prepared")


def run(series, sample, cfg, threads):
    base = os.path.join(SIMULATED, series)
    sdir, rdir = os.path.join(base, "samples"), os.path.join(base, "reads")
    prefix = os.path.join(rdir, "nanosim_" + sample)
    seed = cfg["series"][series]["seed"] * 10 + SAMPLES.index(sample)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=ENV,
               PATH=ENV + ":" + os.environ["PATH"])
    cmd = [sys.executable, patched_simulator(), "metagenome",
           "-gl", os.path.join(sdir, sample + "_genome_list.tsv"),
           "-a", os.path.join(sdir, sample + "_abundance.tsv"),
           "-dl", os.path.join(sdir, sample + "_dna_type_list.tsv"),
           "-c", os.path.join(SIMULATED, cfg["common"]["nanosim_model"]),
           "-o", prefix, "--seed", str(seed), "-t", str(threads)]
    with open(prefix + ".log", "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    merge(prefix, os.path.join(rdir, sample + ".nanosim.fasta"))
    print(series, sample, "done", flush=True)


def merge(prefix, out):
    """Concatenate NanoSim's aligned and unaligned reads (one file per thread) into one fasta and
    remove the per-read error profiles, which are not needed."""
    parts = sorted(glob.glob(prefix + "_sample0_aligned_reads*.fasta")) + \
        sorted(glob.glob(prefix + "_sample0_unaligned_reads*.fasta"))
    if not parts:
        raise FileNotFoundError(prefix)
    with open(out, "w") as handle:
        for path in parts:
            with open(path) as src:
                for line in src:
                    handle.write(line)
    for path in parts + glob.glob(prefix + "_sample0_*error_profile*"):
        os.remove(path)


def summarise(cfg, genomes):
    lengths = {}
    for sp, path in genomes.items():
        lengths[sp] = sum(len(l.strip()) for l in open(path) if not l.startswith(">"))
    for series, sc in cfg["series"].items():
        base = os.path.join(SIMULATED, series)
        target = collections.defaultdict(dict)
        with open(os.path.join(base, "samples", "profile.tsv")) as handle:
            next(handle)
            for line in handle:
                s, st, v = line.rstrip("\n").split("\t")
                target[s][st] = float(v)
        rows = []
        for s in SAMPLES:
            path = os.path.join(base, "reads", s + ".nanosim.fasta")
            if not os.path.exists(path):
                continue
            n, bp = collections.Counter(), collections.Counter()
            name = None
            with open(path) as handle:
                for line in handle:
                    if line.startswith(">"):
                        name = line[1:].split("-", 1)[0]
                        n[name] += 1
                    else:
                        bp[name] += len(line.strip())
            tot_n, tot_bp = sum(n.values()), sum(bp.values())
            for st in sorted(set(n) | set(target[s])):
                sp = st.replace("_variant", "")
                rows.append((s, st, target[s].get(st, 0), 100 * bp[st] / tot_bp, n[st],
                             bp[st] / lengths[sp]))
            rows.append((s, "TOTAL", 100, 100, tot_n, tot_bp / 1e9))
        with open(os.path.join(base, "reads", "reads_summary.tsv"), "w") as handle:
            handle.write("sample\tstrain\ttarget_percent_bases\tachieved_percent_bases\treads\t"
                         "coverage_x (TOTAL: Gbp)\n")
            handle.writelines("{}\t{}\t{:.3f}\t{:.3f}\t{}\t{:.2f}\n".format(*r) for r in rows)
        print("== " + series)
        print(open(os.path.join(base, "reads", "reads_summary.tsv")).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", nargs=2, metavar=("SERIES", "SAMPLE"))
    ap.add_argument("--merge", nargs=2, metavar=("SERIES", "SAMPLE"),
                    help="only merge the per-thread reads of a finished NanoSim run")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--series", nargs="+", help="--prepare: only these series")
    args = ap.parse_args()
    cfg, genomes, dna, shares = load()
    if args.prepare:
        prepare(cfg, genomes, dna, shares, args.series)
    if args.run:
        run(args.run[0], args.run[1], cfg, args.threads)
    if args.merge:
        rdir = os.path.join(SIMULATED, args.merge[0], "reads")
        merge(os.path.join(rdir, "nanosim_" + args.merge[1]),
              os.path.join(rdir, args.merge[1] + ".nanosim.fasta"))
    if args.summarise:
        summarise(cfg, genomes)


if __name__ == "__main__":
    main()
