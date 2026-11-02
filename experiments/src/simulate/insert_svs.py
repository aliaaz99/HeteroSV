"""Insert SVs into each reference, for each SV set.

For every sv set of simulate/series.json (length range) and every reference sequence:
1. SURVIVOR simSV (v1.0.7): 20 indels and 10 tandem duplications in the length range (no
   translocations or inversions), as in rhea's simulation. SURVIVOR is not seeded (it draws from
   the clock), so its outputs are kept as the record of the simulation.
2. 10 complex indels in the same length range, following rhea's custom script
   (insert_mutations.py on its OSF project): a stretch of the variant genome is deleted and a random ACGT
   sequence of independent length is inserted in its place. Differences: positions are drawn with
   a fixed seed, and each complex indel is at least 2 kb from every SURVIVOR event and from the
   other complex indels, so that every SV has a clean ground truth (rhea only spaced the complex
   indels from each other). The final variant header is '<name>-mutated', as in rhea.
   An sv set with "complex_placement": "rhea" places them exactly as rhea's script does instead
   (complex_indels_rhea), and "survivor_from" reuses another set's SURVIVOR events.
3. The sequence of every SV whose abundance changes between T0 and T1 (insertion: inserted
   sequence; deletion: deleted reference sequence; tandem duplication: duplicated unit; complex
   indel: deleted sequence = 'complex loss' and inserted sequence = 'complex gain'), checked
   directly in the final variant genome: inserted and complex-gain sequences present with their
   reference flanks, deletions and complex losses absent with their flanks joined, duplicated
   units present twice in tandem.

  python experiments/src/simulate/insert_svs.py [--sets standard ...]
Writes data/simulated/_svsets/<set>/{variants/*.fasta, survivor/, sv_table.tsv,
truth_sv_sequences.fasta, summary.tsv}.
"""
import argparse
import json
import os
import random
import subprocess

from Bio import SeqIO

from common import SIMULATED, SRC

SURVIVOR = "SURVIVOR"
N_COMPLEX, MARGIN, FLANK = 10, 2000, 40
PARAMS = """PARAMETER FILE: DO JUST MODIFY THE VALUES AND KEEP THE SPACES!
DUPLICATION_minimum_length: {lo}
DUPLICATION_maximum_length: {hi}
DUPLICATION_number: 10
INDEL_minimum_length: {lo}
INDEL_maximum_length: {hi}
INDEL_number: 20
TRANSLOCATION_minimum_length: 1000
TRANSLOCATION_maximum_length: 3000
TRANSLOCATION_number: 0
INVERSION_minimum_length: 600
INVERSION_maximum_length: 800
INVERSION_number: 0
INV_del_minimum_length: 600
INV_del_maximum_length: 800
INV_del_number: 0
INV_dup_minimum_length: 600
INV_dup_maximum_length: 800
INV_dup_number: 0
"""


def read_one(path):
    rec = next(SeqIO.parse(path, "fasta"))
    return rec.id, str(rec.seq).upper()


def parse_vcf(path):
    events = []
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        info = dict(kv.split("=", 1) for kv in f[7].split(";") if "=" in kv)
        events.append(dict(id=f[2], type=info["SVTYPE"], pos=int(f[1]), end=int(info["END"]),
                           svlen=int(info["SVLEN"])))
    return events


def offsets(ref, mut, ins_seqs, ev):
    """0-based (start, stop) of the event in the reference consistent with the mutated genome.
    Shorter flanks are tried for events that SURVIVOR placed next to another event."""
    for flank in (FLANK, 25, 15):
        o = _offsets(ref, mut, ins_seqs, ev, flank)
        if o is not None:
            return o
    return None


def _offsets(ref, mut, ins_seqs, ev, FLANK):
    for ds in (-1, 0, 1, -2):
        for de in (0, -1, 1, -2):
            s, e = ev["pos"] + ds, ev["end"] + de
            left, right = ref[max(0, s - FLANK):s], ref[e:e + FLANK]
            if ev["type"] == "DEL":
                if mut.count(left + right) == 1:
                    return s, e, ref[s:e]
            elif ev["type"] == "DUP":
                unit = ref[s:e]
                if mut.count(left + unit + unit + right) == 1:
                    return s, e, unit
            elif ev["type"] == "INS":
                seq = ins_seqs.get(ev["pos"])
                left = ref[max(0, s - FLANK):s]
                right = ref[s:s + FLANK]
                if seq and mut.count(left + seq + right) == 1:
                    return s, s, seq
    return None


def complex_indels(ref, variant, busy, lo, hi, rng):
    """rhea-style complex indels on the variant, at reference positions away from busy intervals."""
    out, tries = [], 0
    while len(out) < N_COMPLEX and tries < 20000:
        tries += 1
        d = rng.randint(lo, hi)
        s = rng.randint(MARGIN, len(ref) - hi - MARGIN)
        e = s + d
        if any(s - MARGIN < b and a < e + MARGIN for a, b in busy):
            continue
        loss = ref[s:e]
        if ref.count(loss) != 1 or variant.count(loss) != 1:
            continue
        gain = "".join(rng.choice("ACGT") for _ in range(rng.randint(lo, hi)))
        busy.append((s, e))
        out.append(dict(s=s, e=e, loss=loss, gain=gain,
                        left=ref[s - FLANK:s], right=ref[e:e + FLANK]))
    for c in out:  # apply to the variant genome (positions found by exact match)
        p = variant.find(c["loss"])
        variant = variant[:p] + c["gain"] + variant[p + len(c["loss"]):]
    return variant, out, tries


def complex_indels_rhea(variant, lo, hi, rng):
    """Complex indels placed as rhea's own script does (benchmark/insert_mutations.py), for sv sets
    with "complex_placement": "rhea": starts drawn uniformly on the SURVIVOR-mutated genome and
    spaced >= hi only from each other (they may overlap or touch SURVIVOR events); deleted
    stretches of random length in [lo, hi], each replaced by a random sequence of independent
    length in [lo, hi]. Positions (s, e) are in variant coordinates."""
    n, pos, tries = len(variant), [], 0
    while len(pos) < N_COMPLEX:
        tries += 1
        s = rng.randint(0, n - hi)
        if all(abs(s - p[0]) >= hi for p in pos):
            pos.append((s, s + rng.randint(lo, hi)))
    pos.sort()
    gains = ["".join(rng.choice("ACGT") for _ in range(rng.randint(lo, hi))) for _ in pos]
    out = [dict(s=s, e=e, loss=variant[s:e], gain=g, left=variant[max(0, s - FLANK):s],
                right=variant[e:e + FLANK]) for (s, e), g in zip(pos, gains)]
    parts, last = [], 0
    for c in out:
        parts += [variant[last:c["s"]], c["gain"]]
        last = c["e"]
    return "".join(parts) + variant[last:], out, tries


def check(ref, variant, row):
    """The SV is in the final variant genome; copy numbers account for repeated sequences
    (e.g. a deleted stretch with a second copy elsewhere goes from 2 to 1 copies)."""
    t, seq = row["type"], row["seq"]
    n_ref, n_var = ref.count(seq), variant.count(seq)
    flanks = [(row["left"][-k:], row["right"][:k]) for k in (FLANK, 25, 15)]  # adjacent events
    if t in ("insertion", "complex gain"):
        return any(variant.count(l + seq) == 1 or variant.count(seq + r) == 1 for l, r in flanks) \
            and n_var == n_ref + 1
    if t == "deletion":
        return any(variant.count(l + r) == 1 for l, r in flanks) and n_var == n_ref - 1
    if t == "complex loss":
        return n_var == n_ref - 1
    if t == "tandem dup":
        return variant.count(seq + seq) >= 1 and n_var == n_ref + 1
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(SRC, "simulate", "series.json")))
    genomes = [l.rstrip("\n").split("\t") for l in open(os.path.join(SIMULATED, "_inputs",
                                                                      "genome_list.tsv"))]
    for set_name, sv in cfg["sv_sets"].items():
        if args.sets and set_name not in args.sets:
            continue
        lo, hi = sv["length_bp"]
        base = os.path.join(SIMULATED, "_svsets", set_name)
        for sub in ("survivor", "variants"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        # an sv set with "survivor_from" reuses that set's SURVIVOR events (SURVIVOR is not seeded)
        sdir = os.path.join(SIMULATED, "_svsets", sv.get("survivor_from", set_name), "survivor")
        params = os.path.join(sdir, "parameters.txt")
        if not os.path.exists(params):
            with open(params, "w") as handle:
                handle.write(PARAMS.format(lo=lo, hi=hi))
        rows, summary = [], []
        for gi, (species, ref_path) in enumerate(genomes):
            name, ref = read_one(ref_path)
            prefix = os.path.join(sdir, species)
            if not os.path.exists(prefix + ".vcf"):  # SURVIVOR is not seeded: keep its outputs
                with open(prefix + ".log", "w") as log:
                    subprocess.run([SURVIVOR, "simSV", ref_path, params, "0", "0", prefix],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            _, mut = read_one(prefix + ".fasta")
            ins_seqs = {int(r.id.rsplit("_", 1)[1]): str(r.seq).upper()
                        for r in SeqIO.parse(prefix + ".insertions.fa", "fasta")}
            events, busy, n_bad = parse_vcf(prefix + ".vcf"), [], 0
            label = {"INS": "insertion", "DEL": "deletion", "DUP": "tandem dup"}
            for ev in events:
                o = offsets(ref, mut, ins_seqs, ev)
                if o is None:
                    n_bad += 1
                    busy.append((ev["pos"] - 1, ev["end"]))
                    continue
                s, e, seq = o
                busy.append((s, max(e, s + 1)))
                rows.append(dict(species=species, id="{}-{}".format(species, ev["id"]),
                                 type=label[ev["type"]], ref_start=s, ref_end=e, seq=seq,
                                 left=ref[max(0, s - FLANK):s], right=ref[e:e + FLANK]))
            rng = random.Random(sv["seed"] * 100 + gi)
            if sv.get("complex_placement") == "rhea":
                variant, cplx, tries = complex_indels_rhea(mut, lo, hi, rng)
            else:
                variant, cplx, tries = complex_indels(ref, mut, busy, lo, hi, rng)
            for k, c in enumerate(cplx):
                for part, seq in (("complex loss", c["loss"]), ("complex gain", c["gain"])):
                    rows.append(dict(species=species, id="{}-MUT{}".format(species, k), type=part,
                                     ref_start=c["s"], ref_end=c["e"], seq=seq,
                                     left=c["left"], right=c["right"]))
            out = os.path.join(base, "variants", species + ".fasta")
            with open(out, "w") as handle:
                handle.write(">{}-mutated\n".format(name))
                for i in range(0, len(variant), 80):
                    handle.write(variant[i:i + 80] + "\n")
            mine = [r for r in rows if r["species"] == species]
            ok = [check(ref, variant, r) for r in mine]
            for r, good in zip(mine, ok):
                r["verified"] = good
                r["ref_copies"] = ref.count(r["seq"]) if r["seq"] else 0
            counts = {t: sum(r["type"] == t for r in mine) for t in
                      ("insertion", "deletion", "tandem dup", "complex loss", "complex gain")}
            summary.append(dict(species=species, ref_bp=len(ref), variant_bp=len(variant),
                                survivor_events=len(events), survivor_unresolved=n_bad,
                                complex=len(cplx), complex_tries=tries,
                                verified="{}/{}".format(sum(ok), len(ok)), **counts))
            print(set_name, species, summary[-1], flush=True)
        cols = ["species", "id", "type", "ref_start", "ref_end", "verified", "ref_copies"]
        with open(os.path.join(base, "sv_table.tsv"), "w") as handle:
            handle.write("\t".join(cols + ["length"]) + "\n")
            for r in rows:
                handle.write("\t".join(str(r[c]) for c in cols) + "\t{}\n".format(len(r["seq"])))
        with open(os.path.join(base, "truth_sv_sequences.fasta"), "w") as handle:
            for r in rows:
                handle.write(">{}|{}\n{}\n".format(r["id"], r["type"].replace(" ", "_"), r["seq"]))
        keys = list(summary[0])
        with open(os.path.join(base, "summary.tsv"), "w") as handle:
            handle.write("\t".join(keys) + "\n")
            for s in summary:
                handle.write("\t".join(str(s[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
