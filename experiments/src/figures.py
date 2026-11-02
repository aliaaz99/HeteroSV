"""Draw the figures of the paper's main text from the results of the experiments.

  python experiments/src/figures.py fig1 fig2 fig3 fig4 fig5
Writes results/fig1_overview/, results/fig2_heterophily/, ... as PDF and PNG. Needs LaTeX and dvipng.
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import FIG2, FIG3, FIG4, FIG5, RESULTS, TABLE1

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#2a78d6", "#9ec5f4", "#f0efec",
                                                "#f2a3a2", "#e34948", "#a32a2a"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4, "lines.markersize": 4.5,
})
TEXTWIDTH = 6.5  # the paper's text width (inches); paper figures are drawn at their printed size
PAPER_RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{xcolor}",
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "normal",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.handletextpad": 0.4, "axes.titlepad": 4, "lines.markersize": 4,
}


def paper_style(fn):
    """Draw a figure of the paper with PAPER_RC."""
    def wrapped(*args, **kwargs):
        with plt.rc_context(PAPER_RC):
            return fn(*args, **kwargs)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


def save(fig, name: str) -> None:
    os.makedirs(os.path.dirname(name), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig("{}.{}".format(name, ext), facecolor=SURFACE, pad_inches=0.1)
    plt.close(fig)
    print("wrote", name)


def panel_letter(ax, letter: str):
    if plt.rcParams["text.usetex"]:
        ax.text(-0.02, 1.02, r"\textbf{%s}" % letter, transform=ax.transAxes, fontsize=9.5,
                color=INK, ha="right", va="bottom")
    else:
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=INK, ha="right", va="bottom")


def parse_steps(path):
    blocks, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("dataset="):
            cur = dict(re.findall(r"(\w+)=(\S+)", line))
            cur["steps"] = {}
            blocks.append(cur)
        elif line:
            k, v = line.split("\t")
            cur["steps"][k] = float(v)
    return blocks
RHEA_CALLS = os.path.join(FIG5, "rhea_calls_timing.txt")  # speed/rhea_calls_timing.py
HETEROSV_PIPE = os.path.join(FIG5, "heterosv_pipeline_timing.txt")  # speed/heterosv_pipeline_timing.py


def kv_lines(path, prefix):
    """key=value lines of a benchmark output that start with prefix."""
    return [dict(re.findall(r"(\w+)=(\S+)", l)) for l in open(path) if l.startswith(prefix)]
WINS_ROWS = [  # (pattern(s) in wins_vs_rhea/nodes.csv, label, glyph)
    ("false SV calls removed", None, None),
    (["false call: square"], "noise on a square", "square"),
    (["false call: triangle"], "noise on a triangle", "triangle"),
    (["false call: self-loop"], "noise on a self-loop", "loop"),
    ("missed SVs recovered", None, None),
    (["tandem dup: self-loop edge LFC <= 1"], "tandem dup.\\ below the cutoff", "loop"),
    (["indel: two nodes of its triangle beyond 1 SD"], "indel whose flanks also change", "triangle2"),
    (["indel: another node of its triangle is the outlier"], "indel, another node the outlier", "triangle"),
    (["complex: not on a square", "indel: not on a triangle"], "no clean motif", "chain"),
    (["complex: branches change with the same sign"], "branches shift together", "square2"),
]


def wins_glyph(ax, x, y, kind, sx, sy):
    """A tiny motif at (x, y) of the label axis; sx, sy: data units per glyph unit."""
    sv, fl, ec = "#eda100", "#dedcd5", INK2
    if kind.startswith("triangle"):
        pts = {"a": (-1, -0.6), "b": (1, -0.6), "u": (0, 0.7)}
        edges = [("a", "b"), ("a", "u"), ("u", "b")]
        col = {"a": "#f2a3a2" if kind == "triangle2" else fl, "b": "#f2a3a2" if kind == "triangle2" else fl, "u": sv}
    elif kind.startswith("square"):
        pts = {"a": (-1.1, 0), "b": (1.1, 0), "u": (0, 0.75), "w": (0, -0.75)}
        edges = [("a", "u"), ("u", "b"), ("a", "w"), ("w", "b")]
        col = {"a": fl, "b": fl, "u": "#6250d6", "w": "#e34948" if kind == "square" else "#9b92e6"}
    elif kind == "loop":
        pts = {"a": (-1, -0.4), "u": (0.2, -0.4)}
        edges = [("a", "u")]
        col = {"a": fl, "u": "#008300"}
        ax.scatter([x + 0.2 * sx], [y + 0.25 * sy], s=26, facecolor="none", edgecolor=ec, linewidth=0.6, zorder=2)
    else:  # chain
        pts = {"a": (-1.1, 0), "u": (0, 0), "b": (1.1, 0)}
        edges = [("a", "u"), ("u", "b")]
        col = {"a": fl, "u": sv, "b": fl}
    for a, b in edges:
        ax.plot([x + pts[a][0] * sx, x + pts[b][0] * sx], [y + pts[a][1] * sy, y + pts[b][1] * sy], color="#a9a7a0",
                lw=0.7, zorder=1)
    for n, (px, py) in pts.items():
        ax.scatter([x + px * sx], [y + py * sy], s=13, color=col[n], edgecolor=ec, linewidth=0.4, zorder=3)


def _wins_data():
    return pd.read_csv(os.path.join(FIG3, "wins_vs_rhea", "nodes.csv"))


def _panel_wins_bars(axl, axb, D):
    """Rhea's errors that HeteroSV fixes, by motif pattern (label axis axl, bar axis axb)."""
    sv = D.truth != "none"
    n_fc, n_sv = int(((~sv) & (D.rhea != "none")).sum()), int((sv & (D.rhea != D.truth)).sum())
    n_rows = len(WINS_ROWS)
    ys = np.arange(n_rows)[::-1].astype(float)
    for ax in (axl, axb):
        ax.set_ylim(-0.6, n_rows - 0.4)
    axl.set_xlim(0, 1)
    axl.axis("off")
    for y, (pat, lab, glyph) in zip(ys, WINS_ROWS):
        if lab is None:
            tot, fixed = (n_fc, int((D.group == "win: false call removed").sum())) if pat.startswith("false") else \
                (n_sv, int(D.group.isin(["win: missed SV", "win: wrong type"]).sum()))
            axl.text(0.0, y, r"\textbf{%s}" % (pat[0].upper() + pat[1:]), ha="left", va="center", fontsize=7.2)
            axb.text(0, y, "{} of Rhea's {} ({:.0f}\\%)".format(fixed, tot, 100 * fixed / tot), ha="left",
                     va="center", fontsize=6.6, color=INK2)
            continue
        d = D[D.pattern.isin(pat)]
        n, n_pc = len(d), int((d.dataset == "partial_change").sum())
        wins_glyph(axl, 0.07, y, glyph, 0.028, 0.32)
        axl.text(0.15, y, lab, ha="left", va="center", fontsize=6.8)
        axb.barh(y, n - n_pc, height=0.62, color=BLUE)
        axb.barh(y, n_pc, left=n - n_pc, height=0.62, color="#9ec5f4")
        axb.text(n + 1.2, y, str(n), ha="left", va="center", fontsize=6.5, color=INK2)
    axb.set_xlim(0, 75)
    for side in ("left", "right", "top"):
        axb.spines[side].set_visible(False)
    axb.set_yticks([])
    axb.grid(axis="y", visible=False)
    axb.grid(axis="x", visible=True)
    axb.set_xlabel("Nodes HeteroSV gets right")
    axb.legend(handles=[Patch(color=BLUE, label="other datasets"), Patch(color="#9ec5f4", label="partial change")],
               loc="lower right", bbox_to_anchor=(0.9, 0.0), fontsize=6.2, handlelength=1.0, borderaxespad=0.1)


def _panel_wins_why(ax, D):
    """Size of the change of rhea's false calls (removed / kept), true SVs, and tandem duplications."""
    sv = D.truth != "none"
    fc = D[(~sv) & (D.rhea != "none")]
    both_sv = D[(D.group == "both right") & sv]
    td_w = D[D.pattern == "tandem dup: self-loop edge LFC <= 1"]
    td_b = D[(D.group == "both right") & (D.truth == "tandem dup")]
    rows = [("removed", fc[fc.group == "win: false call removed"].abs_contrast, BLUE),
            ("kept", fc[fc.group != "win: false call removed"].abs_contrast, MUTED),
            ("true SVs", both_sv.abs_contrast, MUTED),
            ("recovered", td_w.loop_lfc.abs(), BLUE),
            ("found by both", td_b.loop_lfc.abs(), MUTED)]
    yy = [5.2, 4.4, 3.6, 1.6, 0.8]
    rng = np.random.default_rng(0)
    for y, (lab, v, col) in zip(yy, rows):
        v = np.clip(v.dropna().to_numpy(), 0.01, 30)
        ax.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=4, color=col, alpha=0.55, linewidth=0, zorder=3)
        ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.1, zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.text(0.0115, 6.05, "Rhea's false calls, and true SVs: $|$LFC $-$ neighbors' LFC$|$", fontsize=6.4,
            ha="left", va="center", color=INK)
    ax.text(0.0115, 2.45, "Tandem dup.: $|$self-loop LFC$|$", fontsize=6.4, ha="left", va="center", color=INK)
    ax.plot([1, 1], [0.3, 2.1], color=ORANGE, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(1.12, 2.05, "Rhea's\ncutoff", fontsize=6, color=ORANGE, va="top", ha="left", linespacing=0.95)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 30)
    ax.set_ylim(0.2, 6.5)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Size of the change ($|\\log_2|$ fold change)")
    ax.tick_params(axis="y", length=0)


def _ci(ax, d, xx, color, filled, lab):
    ax.errorbar(xx, d.gain, yerr=[d.gain - d.ci_low, d.ci_high - d.gain], fmt="o", ms=3.6, color=color,
                mfc=color if filled else SURFACE, mec=color, lw=0.9, capsize=1.6, label=lab)


def _dataset_axis(ax):
    x = np.arange(len(SIM_ORDER))
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SIM_LABEL[o] for o in SIM_ORDER], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(SIM_ORDER) - 0.4)
    ax.grid(axis="x", visible=False)


def _panel_gain_rules(ax, boot):
    x = np.arange(len(SIM_ORDER))
    for k, (lname, color, lab) in enumerate((("flank rule 0.5/0.5", BLUE, "flank rule"),
                                             ("rhea's rules", ORANGE, "Rhea's rules"))):
        d = boot[(boot.labeller == lname) & (boot.nodes == "all")].set_index("series").reindex(SIM_ORDER)
        _ci(ax, d, x + (-0.15 if k == 0 else 0.15), color, True, lab)
    ax.set_ylim(-0.005, 0.13)
    ax.set_ylabel("Gain in macro-F1 (95\\% CI)")
    ax.legend(loc="upper left", title="initial labels", title_fontsize=6.5, fontsize=6.5, borderaxespad=0.2)
    _dataset_axis(ax)


def _panel_gain_confidence(ax, boot):
    x = np.arange(len(SIM_ORDER))
    for k, (nodes, filled, lab) in enumerate((("uncertain", True, "25\\% least confident"),
                                               ("confident", False, "75\\% most confident"))):
        d = boot[(boot.labeller == "rhea's rules") & (boot.nodes == nodes)].set_index("series").reindex(SIM_ORDER)
        _ci(ax, d, x + (-0.15 if k == 0 else 0.15), ORANGE, filled, lab)
    ax.set_ylim(-0.03, 0.33)
    ax.set_ylabel("Gain in macro-F1 (95\\% CI)")
    ax.legend(loc="upper left", title="Rhea's calls", title_fontsize=6.5, fontsize=6.5, borderaxespad=0.2)
    _dataset_axis(ax)


def _title(fig, x, y, letter, text):
    fig.text(x, y, r"\textbf{%s}" % letter, fontsize=9.5, ha="left", va="center")
    fig.text(x + 0.022, y, text, fontsize=8, ha="left", va="center")


@paper_style
def fig_r3():
    """Paper result 3: what HeteroSV and its learning step correct (wins_vs_rhea/nodes.csv, bootstrap.csv). (a, b) Rhea's errors that HeteroSV fixes and why;
    (c, d) gain of the learning step over its initial labels, and on Rhea's least / most confident calls."""
    boot = pd.read_csv(os.path.join(TABLE1, "bootstrap.csv"))
    D = _wins_data()
    fig = plt.figure(figsize=(TEXTWIDTH, 4.0))
    axl = fig.add_axes([0.0, 0.565, 0.31, 0.375])
    axb = fig.add_axes([0.31, 0.565, 0.275, 0.375])
    axr = fig.add_axes([0.68, 0.565, 0.31, 0.375])
    _panel_wins_bars(axl, axb, D)
    _panel_wins_why(axr, D)
    _title(fig, 0.003, 0.975, "a", "Rhea's errors that HeteroSV fixes")
    _title(fig, 0.598, 0.975, "b", "Why: changes too small for Rhea's rules")
    axc = fig.add_axes([0.075, 0.115, 0.4, 0.31])
    axd = fig.add_axes([0.585, 0.115, 0.4, 0.31])
    _panel_gain_rules(axc, boot)
    _panel_gain_confidence(axd, boot)
    _title(fig, 0.003, 0.465, "c", "Gain of the learning step over its initial labels")
    _title(fig, 0.518, 0.465, "d", "Gain over Rhea, by Rhea's confidence")
    save(fig, os.path.join(FIG3, "fig3"))
OV_POS = {"A": (0.0, 0.0), "B": (0.8, 0.0), "I": (1.2, 0.62), "C": (1.6, 0.0), "P": (2.25, 0.5),
          "Q": (2.25, -0.5), "D": (2.9, 0.0), "E": (3.6, 0.0)}
OV_EDGES = [("A", "B"), ("B", "I"), ("I", "C"), ("B", "C"), ("C", "P"), ("P", "D"), ("C", "Q"), ("Q", "D"),
            ("D", "E")]
OV_CLASS = {"none": "#dedcd5", "insertion": "#eda100", "complex gain": "#6250d6",
            "complex loss": "#e34948", "tandem dup.": "#008300"}


def ov_graph(ax, face, alpha=None, edgecolor=None, ls=None, r=0.17, loop_color="#a9a7a0", loop_lw=1.0,
             xlim=(-0.35, 3.95), ylim=(-0.85, 1.2)):
    """The toy co-assembly graph of the overview: an insertion triangle (B-I-C), a bubble (C-P-D-Q),
    and a self-loop (E)."""
    from matplotlib.patches import Circle
    for a, b in OV_EDGES:
        (x0, y0), (x1, y1) = OV_POS[a], OV_POS[b]
        ax.plot([x0, x1], [y0, y1], color="#a9a7a0", lw=1.0, zorder=1)
    x, y = OV_POS["E"]
    ax.add_patch(Circle((x, y + 0.3), 0.17, fill=False, ec=loop_color, lw=loop_lw, zorder=1))
    for n, (x, y) in OV_POS.items():
        ax.add_patch(Circle((x, y), r, fc=face.get(n, "#ffffff"), ec=(edgecolor or {}).get(n, INK2),
                            lw=0.8, ls=(ls or {}).get(n, "-"), alpha=(alpha or {}).get(n, 1.0), zorder=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


@paper_style
def fig_overview():
    """Fig. 1: the five steps of HeteroSV on a toy co-assembly graph."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig = plt.figure(figsize=(TEXTWIDTH, 1.8))
    w, gap, x0 = 0.168, 0.034, 0.008
    axes = [fig.add_axes([x0 + k * (w + gap), 0.3, w, 0.5]) for k in range(5)]
    titles = ["Co-assemble, align", "Coverage change", "Initial calls (rules)", "Heterophily-aware GNN",
              "Final SV calls"]
    notes = ["one graph from all samples;\neach sample's coverage\nof every node",
             "an SV changes differently\nfrom its flanks",
             "rules give a call and a\nconfidence (faint = low);\nsome calls are wrong",
             "node, flanks, and branch\nkept separate and compared;\ntrained on the initial calls",
             "calls corrected using\nsimilar nodes\nof the graph"]
    for k, ax in enumerate(axes):
        fig.text(x0 + k * (w + gap) + w / 2, 0.93, r"\textbf{%d}\ \ %s" % (k + 1, titles[k]), ha="center",
                 va="center", fontsize=7.6)
        fig.text(x0 + k * (w + gap) + w / 2, 0.2, notes[k], ha="center", va="center", fontsize=6.2,
                 color=INK2, linespacing=1.05)
        if k < 4:
            xa = x0 + k * (w + gap) + w
            fig.patches.append(FancyArrowPatch((xa + 0.004, 0.55), (xa + gap - 0.004, 0.55),
                                               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
                                               color=INK2, lw=0.9))
    # 1: the graph with the coverage of each node in the two samples
    ax = axes[0]
    ov_graph(ax, {})
    cov = {"A": (0.5, 0.5), "B": (0.55, 0.5), "I": (0.15, 0.6), "C": (0.5, 0.55), "P": (0.15, 0.55),
           "Q": (0.55, 0.15), "D": (0.5, 0.5), "E": (0.3, 0.6)}
    cols = ("#5598e7", "#eb8a55")
    for n, (x, y) in OV_POS.items():
        bx = x - 0.13
        by = y - 0.62 if n == "E" else y + 0.2
        for k, hgt in enumerate(cov[n]):
            ax.add_patch(Rectangle((bx + k * 0.14, by), 0.11, hgt * 0.5, fc=cols[k], ec="none", zorder=2))
    yk = 1.62
    ax.text(-0.3, yk, "coverage in", fontsize=6.0, ha="left", va="center", color=INK2, clip_on=False)
    for k, (lab, xk) in enumerate(((r"$S_t$", 1.75), (r"$S_{t+1}$", 2.75))):
        ax.add_patch(Rectangle((xk, yk - 0.1), 0.14, 0.2, fc=cols[k], ec="none", clip_on=False))
        ax.text(xk + 0.2, yk, lab, fontsize=6.4, ha="left", va="center", color=INK2, clip_on=False)
    # 2: LFC colors, with the self-loop colored by its read support
    ax = axes[1]
    lfc = {"A": 0.05, "B": -0.1, "I": 1.8, "C": 0.1, "P": 1.4, "Q": -1.4, "D": 0.0, "E": 0.9}
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    ov_graph(ax, {n: DIV(norm(v)) for n, v in lfc.items()}, loop_color=DIV(norm(1.6)), loop_lw=1.8)
    cax = fig.add_axes([x0 + 1 * (w + gap) + 0.04, 0.315, w - 0.08, 0.022])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax, orientation="horizontal")
    cb.set_ticks([-2, 0, 2])
    cb.set_ticklabels(["fall", "0", "rise"])
    cb.ax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    # 3: initial calls with confidence; Q missed (borderline), A wrongly called (low confidence)
    ax = axes[2]
    init = {"I": "insertion", "P": "complex gain", "E": "tandem dup.", "A": "complex loss"}
    ov_graph(ax, {n: OV_CLASS[init.get(n, "none")] for n in OV_POS},
             alpha={"A": 0.4, "P": 0.7}, edgecolor={"A": "#e34948", "Q": "#e34948"},
             ls={"A": (0, (2, 1.2)), "Q": (0, (2, 1.2))})
    for n in ("A", "Q"):
        x, y = OV_POS[n]
        ax.text(x, y - 0.33, "?", ha="center", va="top", fontsize=7, color="#e34948")
    # 4: the GNN on the graph: one node, its flanks (1-hop), and the other branch (2-hop)
    ax = axes[3]
    roles = {"P": "#f6c96b", "C": "#9ec5f4", "D": "#9ec5f4", "Q": "#c9c1f2"}
    ov_graph(ax, {n: roles.get(n, "#f4f3ef") for n in OV_POS},
             edgecolor={n: (INK2 if n in roles else "#c3c2b7") for n in OV_POS})
    for txt, (x, y), va, ha in (("node", (2.25, 0.72), "bottom", "center"), ("1-hop", (1.5, -0.2), "top", "right"),
                                ("1-hop", (3.0, -0.2), "top", "left"), ("2-hop", (2.25, -0.72), "top", "center")):
        ax.text(x, y, txt, ha=ha, va=va, fontsize=5.6, color=INK)
    # 5: final calls
    ax = axes[4]
    final = {"I": "insertion", "P": "complex gain", "Q": "complex loss", "E": "tandem dup."}
    ov_graph(ax, {n: OV_CLASS[final.get(n, "none")] for n in OV_POS},
             edgecolor={"A": "#1baf7a", "Q": "#1baf7a"})
    for n in ("A", "Q"):
        x, y = OV_POS[n]
        ax.text(x, y - 0.33, r"$\checkmark$", ha="center", va="top", fontsize=6.5, color="#1baf7a")
    handles = [Patch(fc=c, ec=INK2, lw=0.4, label=l) for l, c in OV_CLASS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=6.2,
               frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    save(fig, os.path.join(RESULTS, "fig1_overview", "fig1"))


@paper_style
def fig_runtime_compact():
    """Main-text runtime figure: graph to labels on copies of the Zymo graph, rhea's published code
    vs HeteroSV on one A100 GPU, with only rhea's last feasible memory and the infeasible size
    annotated, and the cheese series (four steps; cheese_runtime/) as stars (the
    full figure with CPU times is 22b, appendix)."""
    rhea = [d for d in kv_lines(RHEA_CALLS, "rhea_calls")
            if d["dataset"] in ("zymo", "rhea-results-zymo") and "total_s" in d]
    ours = [d for d in kv_lines(HETEROSV_PIPE, "heterosv scale") if "total_s" in d and d["device"] == "cuda"]
    cr = parse_steps(os.path.join(FIG5, "cheese_runtime", "rhea.txt"))[0]
    cheese_rhea = cr["steps"]["init_edge_dicts"] + cr["steps"]["detect_structural_variants"]
    ch = pd.read_csv(os.path.join(FIG5, "cheese_runtime", "heterosv.tsv"), sep="\t")
    cheese_ours = ch[ch.device == "cuda"].one_run.sum()
    fig, ax = plt.subplots(figsize=(0.35 * TEXTWIDTH, 1.6))
    rx, ry = [float(d["nodes"]) for d in rhea], [float(d["total_s"]) for d in rhea]
    ax.plot(rx, ry, color=ORANGE, marker="D", ms=2.6, lw=1.1, label="Rhea (published code)")
    ax.plot([float(t["nodes"]) for t in ours], [float(t["total_s"]) for t in ours], color=BLUE,
            marker="o", ms=2.6, lw=1.1, label="HeteroSV (GPU)")
    n_cheese = int(cr["nodes"])
    ax.plot([n_cheese], [cheese_rhea], marker="*", ms=6.5, color=ORANGE, mec=INK2, mew=0.3, ls="none", zorder=4)
    ax.plot([n_cheese], [cheese_ours], marker="*", ms=6.5, color=BLUE, mec=INK2, mew=0.3, ls="none", zorder=4)
    ax.plot([], [], marker="*", ms=7, color=MUTED, ls="none", label="cheese (4 steps)")
    last = max(rhea, key=lambda d: float(d["nodes"]))
    ax.annotate("{:.0f} GB".format(float(last["peak_rss_gb"])), (float(last["nodes"]), float(last["total_s"])),
                textcoords="offset points", xytext=(3, -2), fontsize=5.5, color=INK2, va="top")
    ax.axvline(74100, color=ORANGE, lw=0.6, ls=(0, (2, 2)))
    ax.text(74100 * 1.2, 1.2e3, "Rhea needs\n$\\approx$0.9 TB\nof memory", fontsize=5.5, color=INK2, va="center",
            linespacing=0.95)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="both", which="major")
    ax.set_xlabel("Graph size (nodes)", fontsize=7)
    ax.set_ylabel("Seconds, graph to labels", fontsize=7)
    ax.tick_params(labelsize=6.3)
    ax.set_ylim(0.3, 3e4)
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.5, borderaxespad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, os.path.join(FIG5, "fig5"))
SIM_ORDER = ["zymo", "replica", "long_sv", "low_depth", "partial_change", "uneven_abundance"]
SIM_LABEL = {"zymo": "Zymo", "replica": "replica", "long_sv": "long SVs", "low_depth": "low depth",
             "partial_change": "partial change", "uneven_abundance": "uneven abund.",
             "hgt_m0": "HGT m0", "hgt_m30": "HGT m30", "cheese": "cheese"}
P1_MODELS = [("LR", "LR [node+edge]"), ("MLP", "MLP [node+edge]"), ("GCN", "GCN [node+edge]"),
             ("GCN-K3", "GCN-K3 [node+edge]"), ("GPR-GNN", "GPR-GNN [node+edge]"),
             ("FAGCN", "FAGCN [node+edge]"), ("GraphSAGE", "SAGE [node+edge]"),
             ("TAGConv", "TAG [node+edge]"), ("H2GCN", "H2GCN [node+edge]")]
P1_GROUPS = [("graph-free", 0, 1), ("averaging", 2, 3), ("heterophily-\noriented", 4, 5),
             ("node kept\nseparate", 6, 8)]


@paper_style
def fig_r1():
    """Paper result 1: heterophily and model comparison on every dataset (heterophily.csv,
    supervised/runs_all.csv, learners.csv). One row: (a) homophily with the
    ground truth, (b) learners trained on the ground truth, (c) learners as Step 4 on the flank
    rule's calls."""
    h = pd.read_csv(os.path.join(FIG2, "heterophily.csv"))
    sup = pd.read_csv(os.path.join(FIG2, "supervised", "runs_all.csv"))
    lr = pd.read_csv(os.path.join(FIG2, "learners.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.9), gridspec_kw=dict(width_ratios=[1.05, 1.75, 1.75]))
    ax = axes[0]
    x = np.arange(len(SIM_ORDER))
    d = h[h.labels == "truth"].set_index("dataset").reindex(SIM_ORDER)
    ax.vlines(x, 0, d.adjusted_h, color=BLUE, lw=1.6, zorder=2)
    ax.scatter(x, d.adjusted_h, s=22, color=BLUE, zorder=3)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.text(len(SIM_ORDER) - 0.5, 0.012, "random labels", fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SIM_LABEL[o] for o in SIM_ORDER], rotation=35, ha="right", fontsize=6.4)
    ax.set_xlim(-0.6, len(SIM_ORDER) - 0.4)
    ax.set_ylim(-0.36, 0.08)
    ax.set_ylabel("Adjusted homophily")
    ax.set_title("Heterophilic labels", loc="left")
    panel_letter(ax, "a")
    ax = axes[1]
    m = sup.groupby(["config", "dataset"]).macro_f1.mean()
    for i, (_, cfg) in enumerate(P1_MODELS):
        v = m[cfg].values
        hero = cfg.startswith("H2GCN")
        lfc = "[lfc]" in cfg
        ax.scatter(i + np.linspace(-0.2, 0.2, len(v)), v, s=9, zorder=3,
                   color=BLUE if hero else MUTED, facecolor=SURFACE if lfc else (BLUE if hero else MUTED),
                   linewidth=0.8)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=INK, lw=0.9, zorder=4)
    short = {"graph-free": "graph-\nfree", "averaging": "aver-\naging", "heterophily-\noriented": "hetero-\nphily",
             "node kept\nseparate": "node kept\nseparate", "LFC only": "LFC\nonly"}
    for name, a0, b0 in P1_GROUPS:
        ax.text((a0 + b0) / 2, 0.29, short[name], fontsize=5.8, color=INK2, ha="center", va="bottom",
                linespacing=0.95)
        if a0 > 0:
            ax.axvline(a0 - 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(range(len(P1_MODELS)))
    ax.set_xticklabels([n for n, _ in P1_MODELS], rotation=35, ha="right", fontsize=6.4)
    ax.set_xlim(-0.6, len(P1_MODELS) - 0.4)
    ax.set_ylim(0.27, 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Graph needed, node kept separate", loc="left")
    panel_letter(ax, "b")
    ax = axes[2]
    g = lr.groupby(["labeller", "learner", "dataset"])[["macro_f1", "initial"]].mean()
    g = (g.macro_f1 - g.initial).rename("gain")
    learners = [("MLP", "MLP"), ("GCN", "GCN"), ("GPR-GNN", "GPR-GNN"), ("FAGCN", "FAGCN"),
                ("GraphSAGE", "GraphSAGE"), ("TAGConv", "TAGConv"), ("H2GCN (HeteroSV)", "H2GCN")]
    groups = [("graph-\nfree", 0, 1), ("aver-\naging", 1, 2), ("hetero-\nphily", 2, 4), ("node kept\nseparate", 4, 7)]
    for i, (ln, _) in enumerate(learners):
        v = g.loc[("flank rule (0.5/0.5)", ln)].values
        hero = ln.startswith("H2GCN")
        ax.scatter(i + np.linspace(-0.2, 0.2, len(v)), v, s=9, color=BLUE if hero else MUTED, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=INK, lw=0.9, zorder=4)
    ax.axhline(0, color=INK2, lw=0.7)
    for name, a0, b0 in groups:
        ax.text((a0 + b0 - 1) / 2, 0.135, name, fontsize=5.5, color=INK2, ha="center", va="top", linespacing=0.95)
        if a0 > 0:
            ax.axvline(a0 - 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(range(len(learners)))
    ax.set_xticklabels([n for _, n in learners], rotation=35, ha="right", fontsize=6.4)
    ax.set_xlim(-0.6, len(learners) - 0.4)
    ax.set_ylim(-0.32, 0.14)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Gain in macro-F1")
    ax.set_title("As Step 4 of HeteroSV", loc="left")
    panel_letter(ax, "c")
    fig.tight_layout(w_pad=0.9)
    save(fig, os.path.join(FIG2, "fig2"))
CHEESE_STEPS = (("cheese_hifi_s0", "2→3"), ("cheese_hifi_s1", "3→4"), ("cheese_hifi_s2", "4→9"),
                ("cheese_hifi_s3", "9→13"), ("cheese_hifi_split", "split"))


def _r4_data():
    """The cheese series (loosen/summary.csv, the steps'
    baseline_support/summary.tsv, ts0/nodes_<dataset>.csv, trajectories, hosts)."""
    S = pd.read_csv(os.path.join(FIG4, "loosen", "summary.csv"))
    real, null = S[S.step != "null control"], S[S.step == "null control"].set_index("caller")
    base = pd.concat([pd.read_csv(os.path.join(FIG4, n, "baseline_support", "summary.tsv"), sep="\t")
                      for n, _ in CHEESE_STEPS[:4]]).groupby("caller")[["calls", "testable", "supported"]].sum()
    rows = [("MAG workflow", int(base.loc["MAG workflow"].supported), int(base.loc["MAG workflow"].testable), None),
            ("Sniffles2", int(base.loc["Sniffles2"].supported), int(base.loc["Sniffles2"].testable), None)]
    for lab, c in (("Rhea", "Rhea"), ("HeteroSV", "HeteroSV on flank 0.5/0.0")):
        d = real[real.caller == c]
        rows.append((lab, int(d.supported.sum()), int(d.testable.sum()), int(null.loc[c].sv_calls)))
    pts = []
    for n, lab in CHEESE_STEPS:
        d = pd.read_csv(os.path.join(FIG4, "ts0", "nodes_{}.csv".format(n)))
        calls = pd.read_csv(os.path.join(FIG4, n, "calls.tsv"), sep="\t")
        pts.append((lab, d[d.with_reads].lfc.abs().median(), int((calls.rhea != "none").sum()),
                    int((d["HeteroSV (T_S=0)"] != "none").sum())))
    P = pd.DataFrame(pts, columns=["step", "change", "Rhea", "HeteroSV"]).sort_values("change")
    # the Psychrobacter strain replacement (trajectories.py, hosts.py)
    T = pd.read_csv(os.path.join(FIG4, "trajectories", "combined.tsv"), sep="\t")
    calls = pd.read_csv(os.path.join(FIG4, "trajectories", "calls.tsv"), sep="\t")
    Bn = pd.read_csv(os.path.join(FIG4, "hosts", "bins.tsv"), sep="\t").set_index("bin")
    d = T.drop_duplicates("node_id").dropna(subset=["share_w2", "share_w13"])
    psy = d[d.species == "s__Psychrobacter sp014861395"]
    # every node with a supported call of either method, by who found it (trajectories.py, hosts.py)
    A = pd.read_csv(os.path.join(FIG4, "trajectories", "trajectories.tsv"), sep="\t").drop_duplicates("node_id")
    hosts = pd.read_csv(os.path.join(FIG4, "hosts", "sv_hosts.tsv"), sep="\t").set_index("node_id")
    het, rhea = set(calls[calls.caller == "HeteroSV"].node_id), set(calls[calls.caller == "Rhea"].node_id)
    A["found"] = np.where(A.node_id.isin(het) & A.node_id.isin(rhea), "both", np.where(A.node_id.isin(het), "HeteroSV", "Rhea"))
    species = hosts.species.fillna(hosts.graph_species)
    A["species"] = A.node_id.map(species)
    A["mge"] = A.node_id.map(hosts.mge)

    def found(nodes):  # all nodes with supported calls, by who found them
        return {"HeteroSV": len(nodes & (het - rhea)), "both": len(nodes & het & rhea), "Rhea": len(nodes & (rhea - het))}
    psy_nodes = set(calls[calls.node_id.map(species) == "s__Psychrobacter sp014861395"].node_id)
    ser_nodes = set(calls[(calls.node_id.map(species) == "s__Serratia proteamaculans") & (calls.step == "9-13")].node_id)
    psy_all = A[A.node_id.isin(psy_nodes)].dropna(subset=["share_w2", "share_w13"])
    ser = A[A.node_id.isin(ser_nodes)].dropna(subset=["share_w9", "share_w13"])
    return dict(rows=rows, P=P, psy=psy, Bn=Bn, rhea_sup=set(calls[calls.caller == "Rhea"].node_id),
                psy_all=psy_all, psy_found=found(psy_nodes), ser=ser, ser_found=found(ser_nodes),
                weeks=[2, 3, 4, 9, 13], W=["w2", "w3", "w4", "w9", "w13"])


def _r4_change(ax, D, title="Calls against the size of the change",
               xlabel="Change of the community at the step (median |LFC| of nodes)", short=False):
    P = D["P"]
    for col, color, mk in (("Rhea", ORANGE, "s"), ("HeteroSV", BLUE, "o")):
        ax.plot(P.change, P[col], color=color, lw=1.2, zorder=2)
        for _, r in P.iterrows():
            filled = r.step != "split"
            ax.plot(r.change, r[col], mk, color=color, mfc=color if filled else SURFACE, mec=color, ms=4.5, zorder=3)
        ax.plot([], [], mk + "-", color=color, label=col, ms=4)
    for _, r in P.iterrows():
        ax.text(r.change, 790 if r.step in ("4→9", "9→13") else 740,
                "control" if r.step == "split" else ("" if short else "wk ") + r.step.replace("→", "$\\to$"),
                ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2])
    ax.set_xticklabels(["0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_ylim(0, 900)
    ax.set_yticks([0, 200, 400, 600])
    ax.set_xlim(0.12, 3.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SV calls")
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_title(title, loc="left")
FOUND = (("both", MUTED, 0.7, 0.55, 2), ("Rhea", ORANGE, 1.1, 0.95, 3), ("HeteroSV", BLUE, 1.0, 0.9, 4))


def _r4_found(ax, T, weeks, title, counts, legend_loc, bbox=None):
    """Allele shares of SV nodes with supported calls, coloured by the method that found them; the legend
    gives the nodes drawn (reads in every week shown) of all nodes with supported calls."""
    shown = T.found.value_counts()
    for found, color, lw, alpha, z in FOUND:
        for r in T[T.found == found].itertuples():
            ax.plot(weeks, [getattr(r, "share_w{}".format(w)) for w in weeks], color=color, lw=lw, alpha=alpha,
                    ls=(0, (3, 1.5)) if r.mge == "plasmid" else "-", zorder=z)
    ax.set_xticks(weeks)
    ax.set_xlim(1.5, 13.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Week of ripening")
    ax.set_ylabel("Share of the SV allele")
    ax.set_title(title, loc="left")
    handles = [Line2D([], [], color=color, lw=1.0, label="{} ({} of {})".format(lab, shown.get(found, 0), counts[found]))
               for found, color, lab in (("HeteroSV", BLUE, "HeteroSV only"), ("Rhea", ORANGE, "Rhea only"),
                                         ("both", MUTED, "both")) if counts[found]]
    handles.append(Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 1.5)), label="on a plasmid"))
    ax.legend(handles=handles, loc=legend_loc, bbox_to_anchor=bbox, fontsize=6, handlelength=1.6,
              labelspacing=0.25, borderaxespad=0.2, frameon=False)


def _r4_psy(ax, D, title="\\emph{Psychrobacter}: strain replacement"):
    _r4_found(ax, D["psy_all"], [2, 3, 4, 9, 13], title, D["psy_found"], "center right")


def _r4_serratia(ax, D, title="\\emph{Serratia}: a rare strain expands"):
    _r4_found(ax, D["ser"], [2, 9, 13], title, D["ser_found"], "center left", bbox=(0.0, 0.43))


@paper_style
def fig_r4():
    """Paper result 4: the cheese series, (a) the Psychrobacter strain replacement and (b) the Serratia
    expansion, each SV node coloured by the method that found it, (c) calls against the size of the change
    (read support and the split-sample control are Table 2)."""
    D = _r4_data()
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15))
    _r4_psy(axs[0], D)
    _r4_serratia(axs[1], D)
    _r4_change(axs[2], D, title="Calls vs.\\ size of the change", xlabel="Change at the step (median |LFC|)",
               short=True)
    for ax, letter in zip(axs, "abc"):
        panel_letter(ax, letter)
    fig.tight_layout(w_pad=0.8)
    save(fig, os.path.join(FIG4, "fig4"))


FIGURES = {"fig1": fig_overview, "fig2": fig_r1, "fig3": fig_r3, "fig4": fig_r4, "fig5": fig_runtime_compact}

if __name__ == "__main__":
    for which in sys.argv[1:]:
        FIGURES[which]()
