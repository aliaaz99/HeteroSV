"""Fig. 2b: learners trained on the ground truth (supervised cross-validation).

Every model sees the same nodes, labels and splits: repeated stratified 5-fold over the nodes
(10 seeds x 5 folds); inside each training fold, a stratified validation split used only for early
stopping of the neural models (LR is fit on train + val); class-weighted losses; features
standardized with train(+val) statistics; scores on the held-out fold only. Graphs are transductive:
test nodes stay in the graph, as in standard node classification.

  python experiments/src/learners/supervised.py [--device cuda:0]
Writes results/fig2_heterophily/supervised/runs_all.csv.
"""
import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import FIG2, load_rhea_graph
from heterosv import prepare, standardise
from heterosv.models import build

DATASETS = ("zymo", "sim_replica", "sim_long_sv", "sim_low_depth", "sim_partial_change", "sim_uneven_abundance")
CONFIGS = {  # name: (model, features)
    "LR [node+edge]": ("lr", "node_edge"), "MLP [node+edge]": ("mlp", "node_edge"),
    "GCN [node+edge]": ("gcn", "node_edge"), "GCN-K3 [node+edge]": ("gcnk", "node_edge"),
    "SAGE [node+edge]": ("sage", "node_edge"), "TAG [node+edge]": ("tag", "node_edge"),
    "GPR-GNN [node+edge]": ("gpr", "node_edge"), "FAGCN [node+edge]": ("fagcn", "node_edge"),
    "H2GCN [node+edge]": ("h2gcn", "node_edge"), "LR [lfc]": ("lr", "lfc"), "LR [node]": ("lr", "node"),
    "MLP [node]": ("mlp", "node"), "H2GCN [lfc]": ("h2gcn", "lfc"), "H2GCN [node]": ("h2gcn", "node")}
LR, WD, EPOCHS, PATIENCE = 0.01, 5e-4, 500, 50


def train_torch(kind, prep, y, n_classes, tr, va, device):
    """Predicted class per node and the epoch used."""
    x = torch.tensor(standardise(prep.x, np.concatenate([tr, va])), dtype=torch.float32, device=device)
    yt = torch.tensor(np.where(y < 0, 0, y), dtype=torch.long, device=device)
    counts = np.bincount(y[tr], minlength=n_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = build(kind, x.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    tr_t, va_t = torch.tensor(tr, device=device), torch.tensor(va, device=device)
    best_key, best_state, best_epoch, wait = None, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, prep.edge_index)[tr_t], yt[tr_t], weight=weight).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, prep.edge_index)
            val_loss = F.cross_entropy(out[va_t], yt[va_t], weight=weight).item()
            val_f1 = f1_score(y[va], out[va_t].argmax(1).cpu().numpy(), average="macro",
                              labels=list(range(n_classes)), zero_division=0)
        key = (round(val_f1, 6), -val_loss)
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, wait = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, prep.edge_index).argmax(1).cpu().numpy(), best_epoch


def train_lr(prep, y, fit_idx, seed):
    x = standardise(prep.x, fit_idx)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    clf.fit(x[fit_idx], y[fit_idx])
    return clf.predict(x)


def scores(y_true, y_pred, n_classes, none_idx):
    labels = list(range(n_classes))
    sv_labels = [k for k in labels if k != none_idx]
    return dict(macro_f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                sv_macro_f1=f1_score(y_true, y_pred, labels=sv_labels, average="macro", zero_division=0),
                sv_detect_f1=f1_score(y_true != none_idx, y_pred != none_idx, zero_division=0),
                accuracy=float((y_true == y_pred).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    device = torch.device(args.device)
    out = os.path.join(FIG2, "supervised")
    os.makedirs(out, exist_ok=True)
    rows = []
    for name in DATASETS:
        g = load_rhea_graph(name, labels="truth")
        labelled = np.where(g.y >= 0)[0]
        k, none = len(g.class_names), g.class_names.index("none")
        prepared = {f: prepare(g, device, features=f) for f in {c[1] for c in CONFIGS.values()}}
        for seed in range(args.seeds):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (a, b) in enumerate(skf.split(labelled, g.y[labelled])):
                trval, te = labelled[a], labelled[b]
                tr, va = train_test_split(trval, test_size=0.2, random_state=seed, stratify=g.y[trval])
                for cname, (kind, feats) in CONFIGS.items():
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    epoch = np.nan
                    if kind == "lr":
                        pred = train_lr(prepared[feats], g.y, trval, seed)
                    else:
                        pred, epoch = train_torch(kind, prepared[feats], g.y, k, tr, va, device)
                    rows.append(dict(dataset=name, config=cname, seed=seed, fold=fold, epochs_used=epoch,
                                     **scores(g.y[te], pred[te], k, none)))
            print(name, "seed", seed, "done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "runs_all.csv"), index=False)


if __name__ == "__main__":
    main()
