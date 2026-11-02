"""Step 3 of the paper: initial calls and confidences from the flank rule."""
import numpy as np
from scipy.stats import rankdata


def weighted_median(v, w):
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o])
    return v[o][np.searchsorted(c, 0.5 * c[-1])]


def covered(g):
    """Nodes with reads in one of the two samples of the step."""
    return g.node_feats[:, 1:3].sum(1) > 0


def sv_score(g):
    """SV score of every node: how much it stands out from its length-weighted neighborhood, and how
    short it is compared with its longest neighbor (Eq. 4 of the paper)."""
    lfc, loglen = g.node_feats[:, 0], g.node_feats[:, 3]
    nbrs = [[] for _ in range(g.n_nodes)]
    for i, j in g.edges["union"].pairs:
        if i != j:
            nbrs[i].append(j)
            nbrs[j].append(i)
    glob = np.median(lfc)
    anchor, short = np.zeros(g.n_nodes), np.zeros(g.n_nodes)
    for u, nb in enumerate(nbrs):
        if not nb:  # isolated: compare with all nodes, no bubble
            anchor[u] = abs(lfc[u] - glob)
            continue
        nb = np.array(nb)
        anchor[u] = abs(lfc[u] - weighted_median(lfc[nb], 10 ** loglen[nb]))
        short[u] = loglen[nb].max() - loglen[u]
    pct = lambda x: rankdata(x) / len(x)
    return np.sqrt(pct(anchor) * pct(short))


def patterns(g):
    """Best local pattern per node: (type name or None, pattern strength; Eqs. 2-3 of the paper)."""
    lfc = g.node_feats[:, 0]
    es = g.edges["union"]
    nbrs = [set() for _ in range(g.n_nodes)]
    loop = np.full(g.n_nodes, np.nan)
    for (i, j), e in zip(es.pairs, es.lfc):
        if i == j:
            loop[i] = e
        else:
            nbrs[i].add(int(j))
            nbrs[j].add(int(i))
    kind = np.array([None] * g.n_nodes, dtype=object)
    score = np.full(g.n_nodes, -np.inf)
    for u in range(g.n_nodes):
        nu = sorted(nbrs[u])
        best = (-np.inf, None)
        for p, a in enumerate(nu):
            for b in nu[p + 1:]:
                m, pen = (lfc[a] + lfc[b]) / 2, abs(lfc[a] - lfc[b]) / 2
                du = lfc[u] - m
                if b in nbrs[a]:
                    s = abs(du) - pen
                    if s > best[0]:
                        best = (s, "insertion" if du > 0 else "deletion")
                for w in nbrs[a] & nbrs[b]:
                    if w == u:
                        continue
                    dw = lfc[w] - m
                    if du * dw < 0:
                        s = min(abs(du), abs(dw)) - pen
                        if s > best[0]:
                            best = (s, "complex gain" if du > 0 else "complex loss")
        if not np.isnan(loop[u]) and abs(loop[u]) > best[0]:
            best = (abs(loop[u]), "tandem dup")
        score[u], kind[u] = best[0], best[1]
    return kind, score


def rank_conf(y, e, none):
    """Within-class percentile rank of evidence e (high e = confident SV, low e = confident none)."""
    r = np.zeros(len(y))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        r[idx] = rankdata(-e[idx] if c == none else e[idx]) / len(idx)
    return r


def ranks(y, e, none, mask):
    """rank_conf restricted to the nodes in mask (zero elsewhere)."""
    r = np.zeros(len(y))
    r[mask] = rank_conf(y[mask], e[mask], none)
    return r


class FlankRule:
    """Precomputed pieces of the flank rule for one graph; thresholds applied on demand."""
    def __init__(self, g):
        names, n = g.class_names, g.n_nodes
        self.none = names.index("none")
        s = sv_score(g)
        kind, self.pscore = patterns(g)
        self.cov = covered(g)
        self.srank = rankdata(s) / n
        pfin = np.where(np.isfinite(self.pscore), self.pscore,
                        np.nanmin(self.pscore[np.isfinite(self.pscore)]) - 1)
        self.e = np.sqrt(self.srank * rankdata(pfin) / n)
        self.kidx = np.array([names.index(t) if t in names else self.none for t in kind])

    def __call__(self, tp, ts):
        """Initial calls at T_P = tp, T_S = ts, and their confidence ranks over all nodes."""
        y = np.where((self.kidx != self.none) & self.cov & (self.pscore >= tp)
                     & (self.srank >= ts), self.kidx, self.none)
        return y, rank_conf(y, self.e, self.none)
