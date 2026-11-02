"""The cheese series: SV allele trajectories against the abundance of their host species (trajectories.py + hosts.py).

For each read-supported HeteroSV call (ts0_vs_rhea.py, trajectories.py), the share of the SV allele A per week (weeks with at
least 5 informative reads) and the abundance of its host bin (hosts.py). A call is a *shift* if the allele
share moves by at least 0.3 between its lowest and highest week; it is *within-species turnover* if,
in addition, the host bin's abundance changes less than 2-fold over the same weeks.

  python experiments/src/cheese/turnover.py
Writes results/fig4_table2_cheese/trajectories/{combined.tsv, summary.md}.
"""
import os

import numpy as np
import pandas as pd

from common import FIG4, md

RES = FIG4
OUT = os.path.join(RES, "trajectories")
WEEKS = ("w2", "w3", "w4", "w9", "w13")
MIN_READS = 5


def main():
    calls = pd.read_csv(os.path.join(OUT, "calls.tsv"), sep="\t")
    calls = calls[calls.caller == "HeteroSV"]
    T = pd.read_csv(os.path.join(OUT, "trajectories.tsv"), sep="\t")
    H = pd.read_csv(os.path.join(RES, "hosts", "sv_hosts.tsv"), sep="\t")
    B = pd.read_csv(os.path.join(RES, "hosts", "bins.tsv"), sep="\t").set_index("bin")
    D = calls.merge(T, on=["node_id", "sv_type"], how="left").merge(H, on="node_id", how="left")
    rows = []
    for r in D.itertuples(index=False):
        sh = {w: getattr(r, "share_" + w) for w in WEEKS if getattr(r, "A_" + w) + getattr(r, "B_" + w) >= MIN_READS}
        sp = r.graph_species if isinstance(r.graph_species, str) else ""
        rec = dict(step=r.step, node_id=r.node_id, sv_type=r.sv_type, species=sp,
                   bin=r.graph_bin if isinstance(r.graph_bin, str) else "", mge=r.mge, n_weeks=len(sh))
        for w in WEEKS:
            rec["share_" + w] = sh.get(w, np.nan)
        if len(sh) >= 2:
            lo, hi = min(sh, key=sh.get), max(sh, key=sh.get)
            rec["share_range"] = sh[hi] - sh[lo]
            if rec["bin"] and rec["bin"] in B.index:
                a, b = B.loc[rec["bin"], lo], B.loc[rec["bin"], hi]
                rec["host_log2fc"] = float(np.log2(max(b, 0.1) / max(a, 0.1)))
        rows.append(rec)
    C = pd.DataFrame(rows)
    C["sweep"] = C.share_range >= 0.3
    ends = C[["share_" + w for w in WEEKS]].apply(lambda v: v.dropna().iloc[-1] if v.notna().any() else np.nan, axis=1)
    starts = C[["share_" + w for w in WEEKS]].apply(lambda v: v.dropna().iloc[0] if v.notna().any() else np.nan, axis=1)
    C["to_fixation"] = (starts.between(0.25, 0.75)) & ((ends >= 0.85) | (ends <= 0.15))
    C["turnover"] = C.sweep & (C.host_log2fc.abs() < 1)
    C.to_csv(os.path.join(OUT, "combined.tsv"), sep="\t", index=False)
    n = len(C)
    traj = C[C.n_weeks >= 3]
    sp = (C[C.species != ""].groupby("species").agg(calls=("node_id", "size"), shifts=("sweep", "sum"),
                                                   to_fixation=("to_fixation", "sum"), turnover=("turnover", "sum")).sort_values("calls", ascending=False))
    text = ("# Cheese: SV allele trajectories and their hosts\n\n"
            "Supported HeteroSV calls: {}; with an allele share in >= 3 weeks: {}; shifts (share range >= 0.3): {}; to near fixation (from 0.25-0.75 to <= 0.15 or >= 0.85): {}; "
            "with a named host: {}; within-species turnover (sweep, host < 2-fold): {}.\n\n".format(
                n, len(traj), int(C.sweep.sum()), int(C.to_fixation.sum()), int((C.species != "").sum()), int(C.turnover.sum()))
            + "By host species:\n\n" + md(sp.reset_index())
            + "\n\nSweeps by mobile-element status:\n\n" + md(C.groupby("mge").agg(calls=("node_id", "size"), sweeps=("sweep", "sum")).reset_index())
            + "\n\nTurnover cases:\n\n" + md(C[C.turnover][["step", "node_id", "sv_type", "species", "mge"] + ["share_" + w for w in WEEKS]
                                            + ["host_log2fc"]].round(2)) + "\n")
    with open(os.path.join(OUT, "summary.md"), "w") as h:
        h.write(text)
    print(text)


if __name__ == "__main__":
    main()
