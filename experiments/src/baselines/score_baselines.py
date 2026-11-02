"""Score the two baseline SV callers as in rhea's paper (per-SV recall and precision).

Callers (data/simulated/<series>/baselines/; per_sample_assembly.sh, mag_workflow.sh,
assembly_comparison.sh, sniffles.sh):
  MAG workflow          MUM&Co on every pair of T0 / T1 MetaBAT bins
  Assembly comparison   MUM&Co on the whole T0 and T1 assemblies
  Sniffles2             reads of both samples aligned to the T0 assembly; an SV is a change if
                        Sniffles2's combined calling genotypes it as present in T1 (an alternative
                        allele) and absent in T0; insertions keep their sequence, deletions and
                        duplications take theirs from the T0 assembly; types are strict (an
                        insertion call does not match a tandem duplication)
Each MUM&Co call (SVs of T1 relative to T0, with the DNA of the event: from the T0 assembly, or
from the T1 assembly for insertions and contractions) becomes a call sequence. The 500 simulated
SV sequences of the series are aligned to the call sequences with minimap2 (-c -N 50, as the ground
truth); an SV sequence is detected if any call of a compatible type has a hit to it, and a
call is correct if it has a hit to an SV sequence of a compatible type. MUM&Co does not call
complex indels, so, as in rhea's evaluation, a deletion may match a complex-loss sequence and an
insertion a complex-gain sequence; duplications match tandem duplications. Contractions,
inversions, and translocations are not simulated and count as false calls.
These callers are not run on Rhea's Zymo data; for Zymo the MAG-workflow counts reported in Rhea's
paper are used (recall only).

  python experiments/src/baselines/score_baselines.py [--control | --sniffles-modes]
--control scores the simulated SV sequences of the replica as if they were calls (recall and
precision must be 1). Writes results/table1_benchmark/baselines/{scores.csv, per_type.csv, summary.md,
calls/}; --sniffles-modes writes sniffles_modes{.csv,_per_type.csv,.md} (Sniffles2 default and mosaic).
"""
import argparse
import glob
import os
import re
import subprocess

import pandas as pd

from common import SIMULATED, SRC, TABLE1, md

OUT = os.path.join(TABLE1, "baselines")
MINIMAP2 = "minimap2"
SERIES = ("replica", "long_sv", "low_depth", "partial_change", "uneven_abundance")
COMPAT = {"insertion": {"insertion", "complex gain"}, "deletion": {"deletion", "complex loss"},
          "duplication": {"tandem dup"}}
TYPES = ("insertion", "deletion", "complex gain", "complex loss", "tandem dup")
# rhea's paper (Curry et al. 2024, Fig. 3b text): MAG workflow on Zymo, even distribution
ZYMO_MAG = {"insertion": (19, 105), "deletion": (23, 95), "complex gain": (0, 100),
            "complex loss": (0, 100), "tandem dup": (25, 100)}


def sv_set(series):
    cfg = __import__("json").load(open(os.path.join(SRC, "simulate", "series.json")))
    return os.path.join(SIMULATED, "_svsets", cfg["series"][series]["sv_set"], "truth_sv_sequences.fasta")


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def mumco_calls(paths):
    """(call id, MUM&Co type, sequence) from SVs_all.withfragment.tsv files."""
    calls = []
    for k, path in enumerate(paths):
        with open(path) as handle:
            next(handle)
            for n, line in enumerate(handle):
                f = line.rstrip("\n").split("\t")
                if len(f) < 6:
                    continue
                seq = f[8] if len(f) > 8 and re.fullmatch(r"[ACGTNacgtn]+", f[8] or "") else ""
                calls.append(("c{}_{}".format(k, n), f[5], seq.upper()))
    return calls


def sniffles_records(vcf):
    """(fields, INFO, per-sample FORMAT) of the records of a Sniffles2 VCF that pass its filters."""
    cols = None
    for line in open(vcf):
        if line.startswith("##"):
            continue
        f = line.rstrip("\n").split("\t")
        if line.startswith("#"):
            cols = f
            continue
        if f[6] not in ("PASS", "."):
            continue
        info = dict(kv.split("=", 1) if "=" in kv else (kv, True) for kv in f[7].split(";"))
        fmt = f[8].split(":")
        yield f, info, {cols[k]: dict(zip(fmt, f[k].split(":"))) for k in range(9, len(f))}


def sniffles_calls(vcf, asm, rule="genotype"):
    """(call id, type, sequence) for the SVs that Sniffles2 genotypes as present in T1 and absent
    in T0 (combined two-sample VCF; records that do not pass Sniffles2's filters are skipped).
    rule="support" instead keeps the SVs that Sniffles2 finds in the reads of T1 and not in those
    of T0 (a sample's ID is not NULL), whatever their genotype, as mosaic SVs are often 0/0.
    rule="per_sample" skips the combined calling, which drops most mosaic SVs: it keeps the calls
    of T1's own VCF (t1.vcf next to vcf) without a T0 call of the same type within 1 kb."""
    if not os.path.exists(vcf):
        return []
    contig = {k.split()[0]: v for k, v in read_fasta(asm).items()}
    names = {"INS": "insertion", "DEL": "deletion", "DUP": "duplication"}
    calls = []

    def present(sample):
        if rule == "support":
            return sample.get("ID", "NULL") != "NULL"
        return "1" in sample.get("GT", "./.").replace("|", "/").split("/")
    if rule == "per_sample":
        w = os.path.dirname(vcf)
        t0 = [(f[0], int(f[1]), info.get("SVTYPE")) for f, info, _ in sniffles_records(os.path.join(w, "t0.vcf"))]
        records = [(f, info) for f, info, _ in sniffles_records(os.path.join(w, "t1.vcf"))
                   if not any(c == f[0] and t == info.get("SVTYPE") and abs(p - int(f[1])) <= 1000
                              for c, p, t in t0)]
    else:
        records = [(f, info) for f, info, sample in sniffles_records(vcf)
                   if present(sample.get("t1", {})) and not present(sample.get("t0", {}))]
    for f, info in records:
        t = info.get("SVTYPE", "")
        pos, end = int(f[1]), int(info.get("END", f[1]))
        if t == "INS":
            seq = f[4] if re.fullmatch(r"[ACGTNacgtn]+", f[4]) else ""
            if seq and f[3] and seq[0].upper() == f[3][0].upper():
                seq = seq[1:]  # drop the padding base shared with REF
        elif t in ("DEL", "DUP"):
            seq = contig.get(f[0], "")[pos:end]  # VCF POS is the base before the event (1-based)
        else:
            seq = ""
        calls.append(("s{}".format(len(calls)), names.get(t, t.lower() or "other"), seq.upper()))
    return calls


def score(series, calls, tag):
    """Per-SV recall per type and call precision for one caller on one series."""
    truth = read_fasta(sv_set(series))
    ttype = {p: p.split("|")[1].replace("_", " ") for p in truth}
    os.makedirs(os.path.join(OUT, "calls"), exist_ok=True)
    fa = os.path.join(OUT, "calls", "{}_{}.fasta".format(tag, series))
    with open(fa, "w") as handle:
        for cid, ctype, seq in calls:
            if seq:
                handle.write(">{}|{}\n{}\n".format(cid, ctype, seq))
    hits = set()
    if os.path.getsize(fa):
        paf = subprocess.run([MINIMAP2, "-c", "-N", "50", "-t", "16", fa, sv_set(series)],
                             capture_output=True, text=True, check=True).stdout
        for line in paf.splitlines():
            f = line.split("\t")
            part, cid, ctype = f[0], f[5].split("|")[0], f[5].split("|")[1]
            if ttype[part] in COMPAT.get(ctype, set()):
                hits.add((part, cid))
    found = {p for p, _ in hits}
    good = {c for _, c in hits}
    per_type = {t: (sum(p in found for p in truth if ttype[p] == t), sum(ttype[p] == t for p in truth))
                for t in TYPES}
    n = len(calls)
    return dict(recall=len(found) / len(truth), precision=len(good) / n if n else float("nan"),
                calls=n, calls_without_sequence=sum(not s for _, _, s in calls),
                other_type_calls=sum(c not in COMPAT for _, c, _ in calls)), per_type


def wall(path):
    """Seconds from a /usr/bin/time -v file or a 'name seconds' line."""
    if not os.path.exists(path):
        return float("nan")
    text = open(path).read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    if m:
        parts = [float(x) for x in m.group(1).split(":")]
        return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
    return float(text.split()[-1])


def sniffles_modes():
    """Sniffles2 in its default mode and in mosaic mode (with and without the SVs above the mosaic
    allele fraction; 19_sniffles.sh SNIFFLES_MODE), each scored with the three rules of
    sniffles_calls."""
    rows, prow = [], []
    for s in SERIES:
        B = os.path.join(SIMULATED, s, "baselines")
        asm = os.path.join(B, "asm_t0", "assembly.fasta")
        for mode, name in (("default", "sniffles"), ("mosaic", "sniffles_mosaic"),
                           ("mosaic_germline", "sniffles_mosaic_germline")):
            vcf = os.path.join(B, name, "merged.vcf")
            if not os.path.exists(vcf):
                continue
            secs = wall(os.path.join(B, "asm_t0.time.txt")) + wall(os.path.join(B, name + "_time.txt"))
            for rule in ("genotype", "support", "per_sample"):
                res, per_type = score(s, sniffles_calls(vcf, asm, rule), "sniffles2-{}-{}".format(mode, rule))
                r, p = res["recall"], res["precision"]
                f1 = 2 * r * p / (r + p) if r + p else 0.0
                rows.append(dict(series=s, mode=mode, rule=rule, seconds_from_reads=secs, f1=f1, **res))
                for t, (k, n) in per_type.items():
                    prow.append(dict(series=s, mode=mode, rule=rule, sv_type=t, detected=k, total=n, recall=k / n))
        print(s, "done", flush=True)
    df, pt = pd.DataFrame(rows), pd.DataFrame(prow)
    df.to_csv(os.path.join(OUT, "sniffles_modes.csv"), index=False)
    pt.to_csv(os.path.join(OUT, "sniffles_modes_per_type.csv"), index=False)
    wide = pt.pivot_table(index=["series", "mode", "rule"], columns="sv_type", values="recall").round(3)
    text = ("# Sniffles2: default and mosaic modes (per-SV recall, precision, F1)\n\n" + md(df.round(3))
            + "\n\n## Recall per SV type\n\n" + md(wide.reset_index()) + "\n")
    with open(os.path.join(OUT, "sniffles_modes.md"), "w") as handle:
        handle.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--sniffles-modes", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.sniffles_modes:
        sniffles_modes()
        return
    if args.control:
        truth = read_fasta(sv_set("replica"))
        back = {"insertion": "insertion", "complex gain": "insertion", "deletion": "deletion",
                "complex loss": "deletion", "tandem dup": "duplication"}
        calls = [("t{}".format(k), back[p.split("|")[1].replace("_", " ")], s)
                 for k, (p, s) in enumerate(truth.items())]
        res, per_type = score("replica", calls, "control")
        print("positive control (replica truth as calls):", res, per_type)
        return
    rows, prow = [], []
    for s in SERIES:
        B = os.path.join(SIMULATED, s, "baselines")
        t_asm = wall(os.path.join(B, "asm_t0.time.txt")) + wall(os.path.join(B, "asm_t1.time.txt"))
        callers = {
            "MAG workflow": (sorted(glob.glob(os.path.join(B, "mag", "*", "pair_output", "pair.SVs_all.withfragment.tsv"))),
                             t_asm + wall(os.path.join(B, "mag_time.txt"))),
            "Assembly comparison": (sorted(glob.glob(os.path.join(B, "whole", "whole_output", "whole.SVs_all.withfragment.tsv"))),
                                    t_asm + wall(os.path.join(B, "whole_time.txt"))),
            "Sniffles2": ([os.path.join(B, "sniffles", "merged.vcf")],
                          wall(os.path.join(B, "asm_t0.time.txt")) + wall(os.path.join(B, "sniffles_time.txt")))}
        for name, (paths, secs) in callers.items():
            if name == "Sniffles2":
                calls = sniffles_calls(paths[0], os.path.join(B, "asm_t0", "assembly.fasta"))
            else:
                calls = mumco_calls(paths)
            res, per_type = score(s, calls, name.split()[0].lower())
            extra = {}
            if name == "MAG workflow":
                extra = dict(bins_t0=len(glob.glob(os.path.join(B, "bins_t0", "*.fa"))),
                             bins_t1=len(glob.glob(os.path.join(B, "bins_t1", "*.fa"))),
                             pairs=sum(1 for _ in open(os.path.join(B, "pairs.tsv"))) if os.path.exists(os.path.join(B, "pairs.tsv")) else 0,
                             pairs_with_calls=len(paths))
            rows.append(dict(series=s, caller=name, seconds_from_reads=secs, **res, **extra))
            for t, (k, n) in per_type.items():
                prow.append(dict(series=s, caller=name, sv_type=t, detected=k, total=n, recall=k / n))
        print(s, "done", flush=True)
    for t, (k, n) in ZYMO_MAG.items():
        prow.append(dict(series="zymo (rhea's paper)", caller="MAG workflow", sv_type=t, detected=k, total=n, recall=k / n))
    k = sum(v[0] for v in ZYMO_MAG.values())
    rows.append(dict(series="zymo (rhea's paper)", caller="MAG workflow", recall=k / 500, calls=float("nan")))
    df, pt = pd.DataFrame(rows), pd.DataFrame(prow)
    df.to_csv(os.path.join(OUT, "scores.csv"), index=False)
    pt.to_csv(os.path.join(OUT, "per_type.csv"), index=False)
    wide = pt.pivot_table(index=["series", "caller"], columns="sv_type", values="recall").round(3)
    text = ("# Baseline SV callers: per-SV recall and precision as in rhea's paper\n\n" + md(df.round(3))
            + "\n\n## Recall per SV type\n\n" + md(wide.reset_index()) + "\n")
    with open(os.path.join(OUT, "summary.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
