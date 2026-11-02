"""Paths and datasets shared by the experiments."""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("HETEROSV_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("HETEROSV_RESULTS", os.path.join(ROOT, "results"))
SRC = os.path.join(ROOT, "experiments", "src")
SIMULATED = os.path.join(DATA, "simulated")
CHEESE = os.path.join(DATA, "cheese")
RHEA_SPARSE = os.path.join(ROOT, "third_party", "rhea", "rhea_sparse.py")

TABLE1 = os.path.join(RESULTS, "table1_benchmark")
FIG2 = os.path.join(RESULTS, "fig2_heterophily")
FIG3 = os.path.join(RESULTS, "fig3_learning_step")
FIG4 = os.path.join(RESULTS, "fig4_table2_cheese")
FIG5 = os.path.join(RESULTS, "fig5_speed")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    edge_files: Tuple[str, ...]  # in series order
    gfa: Optional[str] = "assembly_graph.gfa"
    truth: Optional[str] = None  # node ground-truth labels
    step: int = 0


_NANOSIM = ("edge_coverage-t0.nanosim.tsv", "edge_coverage-t1.nanosim.tsv")
_TRUTH = "truth/node_truth_labels.tsv"
DATASETS = {"zymo": DatasetSpec("zymo", os.path.join(DATA, "zymo"), _NANOSIM, truth=_TRUTH)}
for _s in ("replica", "short_sv", "long_sv", "low_depth", "partial_change", "uneven_abundance"):
    DATASETS["sim_" + _s] = DatasetSpec("sim_" + _s, os.path.join(SIMULATED, _s, "dataset"), _NANOSIM, truth=_TRUTH)
for _k in range(4):  # the cheese series: one folder per step (weeks 2-3, 3-4, 4-9, 9-13)
    DATASETS["cheese_hifi_s{}".format(_k)] = DatasetSpec(
        "cheese_hifi_s{}".format(_k), os.path.join(CHEESE, "dataset_s{}".format(_k)),
        tuple("edge_coverage-w{}.sparse.tsv".format(w) for w in (2, 3, 4, 9, 13)), step=_k)
DATASETS["cheese_hifi_split"] = DatasetSpec(  # split-sample control: week 13 split in two
    "cheese_hifi_split", os.path.join(CHEESE, "split_w13", "dataset_split"),
    ("edge_coverage-t1a.sparse.tsv", "edge_coverage-t1b.sparse.tsv"))


def load_rhea_graph(name, step=None, min_class_size=5, labels="rhea"):
    from heterosv import load_graph  # not in the environments of envs/, which need only the paths above
    spec = DATASETS[name]
    return load_graph(spec.folder, spec.edge_files, step=spec.step if step is None else step,
                      min_class_size=min_class_size, labels=labels, truth=spec.truth, gfa=spec.gfa, name=name)


def md(df: pd.DataFrame, index: bool = False) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    df = df.reset_index() if index else df
    fmt = lambda v: "{:.3f}".format(v) if isinstance(v, float) else str(v)
    rows = [" | ".join(map(str, df.columns)), " | ".join("---" for _ in df.columns)]
    rows += [" | ".join(fmt(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join("| {} |".format(r) for r in rows)
