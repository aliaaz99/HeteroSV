"""Steps 4-5 of the paper: train on the initial calls with a noise-robust, confidence-weighted loss,
then call every node."""
import copy
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from heterosv.graph import EDGE_AGG_FEATURE_NAMES, NODE_FEATURE_NAMES, edge_aggregate_features, to_directed
from heterosv.models import build
from heterosv.rules import covered

HIDDEN, ROUNDS, HOPS, DROPOUT, LR, WD, EPOCHS, PATIENCE = 64, 2, 3, 0.2, 0.01, 5e-4, 500, 50
GAMMA = 0.7  # generalized cross-entropy exponent


@dataclass
class Prepared:
    x: np.ndarray
    feature_names: List[str]
    edge_index: torch.Tensor


def prepare(g, device, features="node_edge"):
    """Node features ('lfc', 'node', or the paper's 14 features 'node_edge') and the union graph."""
    if features == "lfc":
        x, names = g.node_feats[:, :1], NODE_FEATURE_NAMES[:1]
    elif features == "node":
        x, names = g.node_feats, list(NODE_FEATURE_NAMES)
    else:
        x = np.hstack([g.node_feats, edge_aggregate_features(g.n_nodes, g.edges["union"])])
        names = NODE_FEATURE_NAMES + EDGE_AGG_FEATURE_NAMES
    return Prepared(x, names, torch.tensor(to_directed(g.edges["union"]), dtype=torch.long, device=device))


def standardise(x, fit_idx):
    mu, sd = x[fit_idx].mean(0), x[fit_idx].std(0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def node_loss(logits, y, gce, gamma=GAMMA):
    if gce:  # generalized cross-entropy, robust to label noise
        p = torch.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return (1 - p ** gamma) / gamma
    return F.cross_entropy(logits, y, reduction="none")


def train(kind, prep, y_fit, w, tr, va, k, device, gce=True, gamma=GAMMA, hp=None, class_weight=True):
    """Class- and sample-weighted training; epoch chosen by weighted macro-F1 on va.
    gamma: GCE exponent; hp: overrides of hidden / layers / hops / dropout / lr / wd;
    class_weight: balance the classes' (confidence-weighted) masses in the loss."""
    hp = hp or {}
    x = torch.tensor(standardise(prep.x, np.concatenate([tr, va])), dtype=torch.float32, device=device)
    yt = torch.tensor(np.where(y_fit < 0, 0, y_fit), dtype=torch.long, device=device)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    mass = np.bincount(y_fit[tr], weights=w[tr], minlength=k)  # weighted class sizes
    cw = mass.sum() / (k * np.maximum(mass, 1e-6)) if class_weight else np.ones(k)
    cw = torch.tensor(cw, dtype=torch.float32, device=device)
    model = build(kind, x.shape[1], k, hidden=hp.get("hidden", HIDDEN), n_layers=hp.get("layers", ROUNDS),
                  dropout=hp.get("dropout", DROPOUT), hops=hp.get("hops", HOPS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.get("lr", LR), weight_decay=hp.get("wd", WD))
    tr_t, va_t = torch.tensor(tr, device=device), torch.tensor(va, device=device)

    def loss_on(out, idx):
        sw = cw[yt[idx]] * wt[idx]
        return (sw * node_loss(out[idx], yt[idx], gce, gamma)).sum() / sw.sum()

    best, best_state, wait = None, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss_on(model(x, prep.edge_index), tr_t).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, prep.edge_index)
            vl = loss_on(out, va_t).item()
            vf = f1_score(y_fit[va], out[va_t].argmax(1).cpu().numpy(), sample_weight=w[va],
                          labels=list(range(k)), average="macro", zero_division=0)
        key = (round(vf, 6), -vl)
        if best is None or key > best:
            best, best_state, wait = key, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x, prep.edge_index)
    return out.argmax(1).cpu().numpy()


def learn(g, prep, y, r, seed, device):
    """One run of HeteroSV's learning step: H2GCN on the nodes with reads, 20% of them for early
    stopping on the initial calls; nodes without reads are 'none'. Stratified unless a class has
    fewer than 2 nodes."""
    k, none = len(g.class_names), g.class_names.index("none")
    cov = covered(g)
    fit = np.where((g.y >= 0) & cov)[0]
    counts = np.bincount(y[fit], minlength=k)
    strat = y[fit] if counts[counts > 0].min() >= 2 else None
    tr, va = train_test_split(fit, test_size=0.2, random_state=seed, stratify=strat)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pred = train("h2gcn", prep, y, r.astype(np.float64), tr, va, k, device, gce=True)
    pred[~cov] = none
    return pred


def majority(preds, k):
    """Call per node over seeds (ties: the lower class index, 'none' first) and its share of seeds."""
    counts = np.stack([np.bincount(p, minlength=k) for p in np.stack(preds, 1)])
    return counts.argmax(1), counts.max(1) / len(preds)
