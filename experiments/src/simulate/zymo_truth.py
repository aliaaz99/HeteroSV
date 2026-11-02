"""Ground-truth node labels for Rhea's simulated Zymo community.

Truth SVs come from the 'True_positives' sheet of zymo_accuracy.xlsx (SURVIVOR indels and
tandem duplications plus the custom complex indels; REF/ALT sequences). The SV type is taken
from the simulation ID (INS/DEL/DUP/MUT), not the sheet's TYPE column, which was hand-edited
for two indels that rhea called as complex indels.

Each SV contributes the sequence that changes abundance between T0 and T1:
  INS -> ALT (insertion), DEL -> REF (deletion), DUP -> REF (tandem dup),
  MUT -> REF (complex loss) and ALT (complex gain).
These are aligned to every graph node (assembly.fasta) with minimap2. A node gets an SV label
when one SV sequence aligns over >= --min-sv-cov of its length and covers >= --min-node-cov
of the node (the node is that SV allele, not a long node that merely contains it).

The paper's labels (in the data archive) use minimap2 2.31; the same steps with minimap2 2.24, the
version of the rest of the pipeline, label slightly more nodes (362 against 351).

  python experiments/src/simulate/zymo_truth.py [--minimap2 minimap2]
Reads data/zymo/{zymo_accuracy.xlsx, assembly.fasta}. Writes data/zymo/truth/.
"""
import argparse
import os
import subprocess

import pandas as pd

from common import DATA

ZYMO_DIR = os.path.join(DATA, "zymo")
PAF_COLS = ["sv_part", "sv_len", "sv_start", "sv_end", "strand", "node", "node_len",
            "node_start", "node_end", "matches", "aln_len", "mapq"]
PARTS = {"INS": [("insertion", "ALT")], "DEL": [("deletion", "REF")],
         "DUP": [("tandem dup", "REF")],
         "MUT": [("complex loss", "REF"), ("complex gain", "ALT")]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minimap2", default="minimap2")
    parser.add_argument("--zymo-dir", default=ZYMO_DIR)
    parser.add_argument("--min-sv-cov", type=float, default=0.8)
    parser.add_argument("--min-node-cov", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    out_dir = os.path.join(args.zymo_dir, "truth")
    os.makedirs(out_dir, exist_ok=True)
    sheet = pd.read_excel(os.path.join(args.zymo_dir, "zymo_accuracy.xlsx"),
                          sheet_name="True_positives")
    fasta = os.path.join(out_dir, "truth_sv_sequences.fasta")
    with open(fasta, "w") as handle:
        for _, row in sheet.iterrows():
            for label, column in PARTS[row["ID"][:3]]:
                handle.write(">{}|{}\n{}\n".format(row["Full ID"], label.replace(" ", "_"),
                                                   row[column]))

    paf = os.path.join(out_dir, "truth_vs_nodes.paf")
    with open(paf, "w") as handle:
        subprocess.run([args.minimap2, "-c", "-N", "50", "-t", str(args.threads),
                        os.path.join(args.zymo_dir, "assembly.fasta"), fasta],
                       stdout=handle, stderr=subprocess.DEVNULL, check=True)

    hits = pd.read_csv(paf, sep="\t", header=None, usecols=range(12), names=PAF_COLS)
    hits["sv_type"] = hits["sv_part"].str.split("|").str[1].str.replace("_", " ")
    hits["node_id"] = hits["node"].str.split("_").str[-1].astype(int)
    hits["sv_cov"] = (hits["sv_end"] - hits["sv_start"]) / hits["sv_len"]
    hits["node_cov"] = (hits["node_end"] - hits["node_start"]) / hits["node_len"]
    hits[["sv_part", "sv_type", "node_id", "sv_len", "node_len", "sv_cov", "node_cov"]].to_csv(
        os.path.join(out_dir, "truth_hits.tsv"), sep="\t", index=False)

    strong = hits[(hits["sv_cov"] >= args.min_sv_cov) & (hits["node_cov"] >= args.min_node_cov)]
    strong = strong.sort_values("node_cov", ascending=False)
    n_conflict = int((strong.groupby("node_id")["sv_type"].nunique() > 1).sum())
    labels = strong.drop_duplicates("node_id")[["node_id", "sv_type", "sv_part", "sv_cov", "node_cov"]]
    labels = labels.rename(columns={"sv_type": "truth_label"}).sort_values("node_id")
    labels.to_csv(os.path.join(out_dir, "node_truth_labels.tsv"), sep="\t", index=False)

    n_parts = sum(len(PARTS[i[:3]]) for i in sheet["ID"])
    print("SV sequences: {} | with any node hit: {} | labelled nodes: {} "
          "(nodes matching >1 SV type: {}, best node_cov kept)".format(
              n_parts, hits["sv_part"].nunique(), len(labels), n_conflict))
    print(labels["truth_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
