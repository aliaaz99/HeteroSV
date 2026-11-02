"""H2GCN, the learner of HeteroSV, and the learners it is compared with in the paper."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv, GCNConv, SAGEConv, SGConv, TAGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, n_classes, dropout):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), self.dropout, self.training)
        return self.out(x)


class _ConvNet(nn.Module):
    """A stack of graph convolutions with ReLU and dropout, then a linear output layer."""
    def __init__(self, convs, hidden, n_classes, dropout):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.out = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, self.training)
        return self.out(x)


def _dims(in_dim, hidden, n_layers):
    dims = [in_dim] + [hidden] * n_layers
    return list(zip(dims[:-1], dims[1:]))


def TAGNet(in_dim, hidden, n_layers, n_classes, dropout, hops):
    return _ConvNet([TAGConv(a, b, K=hops) for a, b in _dims(in_dim, hidden, n_layers)], hidden, n_classes, dropout)


def GCNNet(in_dim, hidden, n_layers, n_classes, dropout):
    return _ConvNet([GCNConv(a, b) for a, b in _dims(in_dim, hidden, n_layers)], hidden, n_classes, dropout)


def GCNKNet(in_dim, hidden, n_layers, n_classes, dropout, hops):
    """GCN with TAG's receptive field: each layer propagates K hops (A~^K h W, SGC-style)."""
    return _ConvNet([SGConv(a, b, K=hops) for a, b in _dims(in_dim, hidden, n_layers)], hidden, n_classes, dropout)


def SAGENet(in_dim, hidden, n_layers, n_classes, dropout):
    return _ConvNet([SAGEConv(a, b) for a, b in _dims(in_dim, hidden, n_layers)], hidden, n_classes, dropout)


def _sym_norm(a):
    d = np.asarray(a.sum(1)).ravel()
    inv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    return sp.diags(inv) @ a @ sp.diags(inv)


def _to_torch(a, device):
    a = a.tocoo()
    return torch.sparse_coo_tensor(np.vstack([a.row, a.col]), a.data, a.shape,
                                   dtype=torch.float32, device=device).coalesce()


class GPRNet(nn.Module):
    """GPR-GNN (Chien et al. 2021): a two-layer MLP, then sum_k gamma_k A~^k Z with learnable
    gamma_k (initialised as personalised PageRank, alpha = 0.1), which may become negative
    (high-pass). A~: symmetric-normalised adjacency with self-loops."""
    def __init__(self, in_dim, hidden, n_layers, n_classes, dropout, hops=10, alpha=0.1):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, n_classes)
        gamma = alpha * (1 - alpha) ** np.arange(hops + 1)
        gamma[-1] = (1 - alpha) ** hops
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.dropout, self.hops, self._a = dropout, hops, None

    def forward(self, x, edge_index):
        if self._a is None:
            ei, w = gcn_norm(edge_index, None, x.shape[0], add_self_loops=True)
            self._a = torch.sparse_coo_tensor(torch.stack([ei[1], ei[0]]), w,
                                              (x.shape[0], x.shape[0])).coalesce()
        x = F.dropout(x, self.dropout, self.training)
        z = self.lin2(F.dropout(F.relu(self.lin1(x)), self.dropout, self.training))
        out = self.gamma[0] * z
        for k in range(1, self.hops + 1):
            z = torch.sparse.mm(self._a, z)
            out = out + self.gamma[k] * z
        return out


class FAGCNNet(nn.Module):
    """FAGCN (Bo et al. 2021): signed edge attention in [-1, 1] passes low- or high-frequency
    signal per edge; eps times the initial representation is added at every layer."""
    def __init__(self, in_dim, hidden, n_layers, n_classes, dropout, eps=0.3):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(FAConv(hidden, eps=eps, dropout=dropout) for _ in range(n_layers))
        self.out = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        h = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        h0 = h
        for conv in self.convs:
            h = conv(h, h0, edge_index)
        return self.out(F.dropout(h, self.dropout, self.training))


class H2GCNNet(nn.Module):
    """H2GCN (Zhu et al. 2020): ego embedding, then n_layers rounds that aggregate the 1-hop and
    the strict 2-hop neighbourhood separately (no self-loops); all rounds are concatenated."""
    def __init__(self, in_dim, hidden, n_layers, n_classes, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.rounds = n_layers
        self.out = nn.Linear(hidden * (2 ** (n_layers + 1) - 1), n_classes)
        self.dropout, self._ops = dropout, None

    def operators(self, edge_index, n):
        src, dst = edge_index.cpu().numpy()
        keep = src != dst
        a = sp.coo_matrix((np.ones(keep.sum()), (dst[keep], src[keep])), shape=(n, n)).tocsr()
        a = ((a + a.T) > 0).astype(np.float64)
        a2 = ((a @ a) > 0).astype(np.float64)
        a2 = (a2 - a2.multiply(a)).tolil()
        a2.setdiag(0)
        a2 = a2.tocsr()
        a2.eliminate_zeros()
        dev = edge_index.device
        return _to_torch(_sym_norm(a), dev), _to_torch(_sym_norm(a2), dev)

    def forward(self, x, edge_index):
        if self._ops is None:
            self._ops = self.operators(edge_index, x.shape[0])
        a1, a2 = self._ops
        r = F.relu(self.embed(x))
        reps = [r]
        for _ in range(self.rounds):
            r = torch.cat([torch.sparse.mm(a1, r), torch.sparse.mm(a2, r)], dim=1)
            reps.append(r)
        return self.out(F.dropout(torch.cat(reps, dim=1), self.dropout, self.training))


def build(kind, in_dim, n_classes, hidden=64, n_layers=2, dropout=0.2, hops=3):
    """A learner by name: mlp, gcn, gcnk, sage, tag, gpr, fagcn, h2gcn."""
    common = dict(hidden=hidden, n_layers=n_layers, n_classes=n_classes, dropout=dropout)
    if kind == "tag":
        return TAGNet(in_dim, hops=hops, **common)
    if kind == "gcnk":
        return GCNKNet(in_dim, hops=hops, **common)
    return {"mlp": MLP, "gcn": GCNNet, "sage": SAGENet, "gpr": GPRNet, "fagcn": FAGCNNet,
            "h2gcn": H2GCNNet}[kind](in_dim, **common)
