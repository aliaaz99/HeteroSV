"""Read a co-assembly graph with per-sample coverage into arrays (Steps 1-2 of the paper).

A dataset folder holds the files Rhea writes for a series (third_party/rhea):
  node_coverage_norm.csv         normalized coverage and log fold change (LFC) of every node
  edge_coverage-<sample>.tsv     N x N read-path counts per sample (Rhea's pseudocount of 1 included)
  edge_coverage-<sample>.sparse.tsv   the same as a table of observed entries (node_a, node_b, count)
  structural_variants-c<step>.tsv or structual_variants.tsv   Rhea's calls
  assembly_graph.gfa             the co-assembly graph (optional)

The graph is the read-supported one: an edge joins two nodes if a read of any sample of the series
walks between them.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Rhea's SV names (old and current) -> class names
SV_CLASS = {
    "insertion": "insertion",
    "deletion": "deletion",
    "mutation win": "complex gain",
    "complex insertion": "complex gain",
    "mutation loss": "complex loss",
    "complex deletion": "complex loss",
    "tandem duplication gain": "tandem dup",
    "tandem duplication loss": "tandem dup",
}
CLASS_ORDER = ["none", "insertion", "deletion", "complex gain", "complex loss", "tandem dup"]
NODE_FEATURE_NAMES = ["lfc", "log_cov_t0", "log_cov_t1", "log10_length"]
EDGE_AGG_FEATURE_NAMES = [
    "degree", "degree_t0", "degree_t1", "n_t0_only_edges", "n_t1_only_edges",
    "mean_edge_lfc", "min_edge_lfc", "max_edge_lfc", "has_self_loop", "self_loop_lfc",
]


@dataclass
class EdgeSet:
    """Undirected edges (i <= j, 0-based row indices) with per-edge raw counts."""
    pairs: np.ndarray            # (E, 2)
    c0: np.ndarray               # (E,) raw read count in the earlier sample
    c1: np.ndarray               # (E,) raw read count in the later sample
    multiplicity: Optional[np.ndarray] = None  # GFA link count, gfa graph only

    @property
    def lfc(self) -> np.ndarray:
        return np.log2((self.c1 + 1.0) / (self.c0 + 1.0))


@dataclass
class RheaGraph:
    dataset: str
    step: int
    node_ids: np.ndarray
    node_feats: np.ndarray                 # (N, 4) NODE_FEATURE_NAMES
    edges: Dict[str, EdgeSet]              # graph variant -> edges
    y: np.ndarray                          # (N,) class index, -1 = excluded
    class_names: List[str]
    rhea_pred: np.ndarray                  # (N,) Rhea's own call as a class index
    folder: str = ""
    label_source: str = "rhea"
    label_notes: List[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


def _read_count_matrix(path: str, node_ids: np.ndarray) -> np.ndarray:
    if path.endswith(".sparse.tsv"):
        return _read_count_sparse(path, node_ids)
    df = pd.read_csv(path, sep="\t", index_col=0)
    if not (np.array_equal(df.index.to_numpy(dtype=int), node_ids)
            and np.array_equal(df.columns.astype(int).to_numpy(), node_ids)):
        raise ValueError("Row/column order of {} does not match node_coverage node_ids".format(path))
    counts = df.to_numpy(dtype=np.float64) - 1.0  # remove Rhea's pseudocount
    if counts.min() < 0:
        raise ValueError("Negative counts after removing pseudocount in {}".format(path))
    return counts


def _read_count_sparse(path: str, node_ids: np.ndarray) -> sp.csr_matrix:
    """Edge counts written by third_party/rhea/rhea_sparse.py as a sparse N x N matrix of raw counts."""
    df = pd.read_csv(path, sep="\t")
    row = pd.Series(np.arange(len(node_ids)), index=node_ids)
    i, j = row.loc[df.node_a].to_numpy(), row.loc[df.node_b].to_numpy()
    counts = df["count"].to_numpy(dtype=np.float64) - 1.0  # remove Rhea's pseudocount
    if counts.size and counts.min() < 1:
        raise ValueError("Observed entries without reads in {}".format(path))
    return sp.csr_matrix((counts, (i, j)), shape=(len(node_ids), len(node_ids)))


def _pick(m, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Entries (i, j) of a dense or sparse matrix as a 1-d array."""
    return np.asarray(m[i, j]).ravel() if sp.issparse(m) else m[i, j]


def _coverage_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns[3:]
            if not c.endswith("_colour") and c != "colour"
            and not c.startswith(("linear_change", "log_fold_change"))]


def _union_edges(m0, m1, support) -> EdgeSet:
    """Edges of the read-supported graph; support: boolean matrix of pairs with reads in any sample."""
    if sp.issparse(m0):
        present = sp.triu(support + support.T).tocoo()
        order = np.lexsort((present.col, present.row))  # row-major, as np.nonzero
        i, j = present.row[order].astype(np.int64), present.col[order].astype(np.int64)
        return EdgeSet(pairs=np.stack([i, j], 1), c0=_pick(m0, i, j), c1=_pick(m1, i, j))
    i, j = np.nonzero(np.triu(support | support.T))
    return EdgeSet(pairs=np.stack([i, j], 1), c0=m0[i, j], c1=m1[i, j])


def _gfa_edges(path: str, id_to_row: Dict[int, int], m0, m1) -> EdgeSet:
    mult: Dict[Tuple[int, int], int] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("L"):
                continue
            fields = line.split("\t")
            a = id_to_row[int(fields[1].split("_")[-1])]
            b = id_to_row[int(fields[3].split("_")[-1])]
            key = (min(a, b), max(a, b))
            mult[key] = mult.get(key, 0) + 1
    pairs = np.array(sorted(mult), dtype=np.int64).reshape(-1, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    return EdgeSet(pairs=pairs, c0=_pick(m0, i, j), c1=_pick(m1, i, j),
                   multiplicity=np.array([mult[tuple(p)] for p in pairs], dtype=np.float64))


def _sv_table(folder: str, step: int) -> pd.DataFrame:
    """Rhea's table of SV calls for a step (Rhea's stored Zymo folder keeps the older file name)."""
    for name in ("structual_variants.tsv", "structural_variants-c{}.tsv".format(step)):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return pd.read_csv(path, sep="\t")
    raise FileNotFoundError("No table of Rhea's SV calls in " + folder)


def _sv_column(sv: pd.DataFrame, step: int) -> str:
    """Column prefix of a step: 't<step>-' in Rhea's stored Zymo table, 'c<step>-' otherwise."""
    return "t{}-".format(step) if "t{}-SV_type".format(step) in sv.columns else "c{}-".format(step)


def _rhea_calls(folder: str, node_ids: np.ndarray, step: int) -> Tuple[np.ndarray, List[str]]:
    """Rhea's SV call per node as a class name ('none' if uncalled); first call kept."""
    sv = _sv_table(folder, step)
    col = _sv_column(sv, step) + "SV_type"
    notes = []
    n_multi = int(sv.dropna(subset=[col]).duplicated("node_id").sum())
    if n_multi:
        notes.append("{} extra SV rows on already-labelled nodes; first call per node kept".format(n_multi))
    first = sv.drop_duplicates("node_id", keep="first").set_index("node_id")[col]
    return first.reindex(node_ids).map(SV_CLASS).fillna("none").to_numpy(), notes


def _encode(names: np.ndarray, min_class_size: int) -> Tuple[np.ndarray, List[str], List[str]]:
    notes = []
    counts = pd.Series(names).value_counts()
    kept = [c for c in CLASS_ORDER if counts.get(c, 0) >= min_class_size]
    dropped = {c: int(counts[c]) for c in counts.index if c not in kept}
    if dropped:
        notes.append("classes with < {} nodes kept in the graph but excluded from "
                     "training/scoring: {}".format(min_class_size, dropped))
    index = {c: k for k, c in enumerate(kept)}
    y = np.array([index.get(n, -1) for n in names], dtype=np.int64)
    return y, kept, notes


def load_graph(folder: str, edge_files: Sequence[str], step: int = 0, min_class_size: int = 0,
               labels: str = "rhea", truth: Optional[str] = None, gfa: Optional[str] = None,
               name: str = "") -> RheaGraph:
    """One step (pair of consecutive samples) of a series.

    labels: 'rhea' (Rhea's calls) or 'truth' (the node labels in the file `truth`, for simulated
    data). HeteroSV itself never uses these labels: with min_class_size=0 every class is kept, and
    its initial calls come from the flank rule (heterosv.rules)."""
    if step + 1 >= len(edge_files):
        raise ValueError("{} has only {} samples".format(folder, len(edge_files)))
    cov = pd.read_csv(os.path.join(folder, "node_coverage_norm.csv"))
    node_ids = cov["node_id"].to_numpy(dtype=int)
    samples = _coverage_columns(cov)
    cov_t0 = cov[samples[step]].fillna(0).to_numpy(dtype=np.float64)
    cov_t1 = cov[samples[step + 1]].fillna(0).to_numpy(dtype=np.float64)
    node_feats = np.stack([
        cov["log_fold_change_{}".format(step)].to_numpy(dtype=np.float64),
        np.log1p(cov_t0),
        np.log1p(cov_t1),
        np.log10(cov["node_length"].to_numpy(dtype=np.float64)),
    ], axis=1)

    m0 = _read_count_matrix(os.path.join(folder, edge_files[step]), node_ids)
    m1 = _read_count_matrix(os.path.join(folder, edge_files[step + 1]), node_ids)
    # The co-assembly graph is built from every sample, so edges with reads in any sample count.
    if sp.issparse(m0):
        support = (m0 > 0) + (m1 > 0)
        for k, edge_file in enumerate(edge_files):
            if k not in (step, step + 1):
                support = support + (_read_count_matrix(os.path.join(folder, edge_file), node_ids) > 0)
    else:
        support = (m0 > 0) | (m1 > 0)
        for k, edge_file in enumerate(edge_files):
            if k not in (step, step + 1):
                support |= _read_count_matrix(os.path.join(folder, edge_file), node_ids) > 0
    edges = {"union": _union_edges(m0, m1, support)}
    if gfa:
        id_to_row = {int(n): k for k, n in enumerate(node_ids)}
        edges["gfa"] = _gfa_edges(os.path.join(folder, gfa), id_to_row, m0, m1)

    rhea_names, notes = _rhea_calls(folder, node_ids, step)
    if labels == "truth":
        if truth is None:
            raise ValueError("No ground-truth labels for " + folder)
        table = pd.read_csv(os.path.join(folder, truth), sep="\t").set_index("node_id")["truth_label"]
        names = table.reindex(node_ids).fillna("none").to_numpy()
    else:
        names = rhea_names
    y, class_names, enc_notes = _encode(names, min_class_size)
    none_idx = class_names.index("none")
    rhea_pred = np.array([class_names.index(n) if n in class_names else none_idx
                          for n in rhea_names], dtype=np.int64)
    return RheaGraph(name or folder, step, node_ids, node_feats, edges, y, class_names, rhea_pred,
                     folder, labels, notes + enc_notes)


def edge_aggregate_features(n_nodes: int, es: EdgeSet) -> np.ndarray:
    """Per-node summaries of incident edges (EDGE_AGG_FEATURE_NAMES)."""
    out = np.zeros((n_nodes, len(EDGE_AGG_FEATURE_NAMES)))
    lfc = es.lfc
    incident: List[List[int]] = [[] for _ in range(n_nodes)]
    for k, (i, j) in enumerate(es.pairs):
        if i == j:
            out[i, 8] = 1.0
            out[i, 9] = lfc[k]
        else:
            incident[i].append(k)
            incident[j].append(k)
    for node, ks in enumerate(incident):
        if not ks:
            continue
        ks = np.array(ks)
        c0, c1, l = es.c0[ks], es.c1[ks], lfc[ks]
        out[node, :8] = [len(ks), (c0 > 0).sum(), (c1 > 0).sum(),
                         ((c0 > 0) & (c1 == 0)).sum(), ((c1 > 0) & (c0 == 0)).sum(),
                         l.mean(), l.min(), l.max()]
    return out


def to_directed(es: EdgeSet) -> np.ndarray:
    """Both directions for every non-self edge, once for self-loops."""
    i, j = es.pairs[:, 0], es.pairs[:, 1]
    off = i != j
    return np.stack([np.concatenate([i, j[off]]), np.concatenate([j, i[off]])])


def rhea_call_confidence(g: RheaGraph) -> np.ndarray:
    """How strongly Rhea's own evidence supports its call on each node (higher = more likely), from
    Rhea's output and the node and edge LFCs only. Comparable only within one called class:
      insertion / deletion  |LFC - median LFC| of the node in the triangle Rhea reported
      complex gain / loss   the same within the reported 4-cycle
      tandem dup            largest |edge LFC| on the node's edges (self-loop included)
      none                  minus |LFC - median LFC of its graph neighbours| (isolated: of all nodes)
    A node whose Rhea rows at this step disagree on the class is ranked below every node of its
    class: the call is ambiguous."""
    sv = _sv_table(g.folder, g.step)
    prefix = _sv_column(sv, g.step)
    sv = sv.dropna(subset=[prefix + "SV_type"])
    row_of = {int(n): k for k, n in enumerate(g.node_ids)}
    lfc = g.node_feats[:, 0]
    es = g.edges["union"]
    max_edge = np.zeros(g.n_nodes)
    nbrs: List[List[int]] = [[] for _ in range(g.n_nodes)]
    for (i, j), e in zip(es.pairs, np.abs(es.lfc)):
        max_edge[i] = max(max_edge[i], e)
        max_edge[j] = max(max_edge[j], e)
        if i != j:
            nbrs[i].append(int(j))
            nbrs[j].append(int(i))
    conf = np.array([-abs(lfc[u] - np.median(lfc[nb])) if nb else -abs(lfc[u] - np.median(lfc))
                     for u, nb in enumerate(nbrs)])
    ambiguous = np.zeros(g.n_nodes, dtype=bool)
    for node_id, rows in sv.groupby("node_id", sort=False):
        u = row_of[int(node_id)]
        classes = rows[prefix + "SV_type"].map(SV_CLASS)
        first = rows.iloc[0]
        cls = classes.iloc[0]
        ambiguous[u] = classes.nunique() > 1
        if cls == "tandem dup":
            conf[u] = max_edge[u]
            continue
        cols = ["neighbor1", "neighbor2"] + (["replacement"] if cls.startswith("complex") else [])
        cycle = [u] + [row_of[int(first[prefix + c])] for c in cols if pd.notna(first[prefix + c])]
        conf[u] = abs(lfc[u] - np.median(lfc[cycle]))
    return np.where(ambiguous, conf - 1e6, conf)
