"""Shared style for every generated manuscript figure.

Single source of truth for palette, sizes and output paths.  Import it; never
hard-code a colour or a font size in an individual figure script.

Palette: Okabe--Ito, validated for CVD separation (worst adjacent pair
Delta-E 11.0 deutan / 25.8 normal vision, all slots inside the lightness band
and above the chroma floor and the 3:1 contrast floor).  Every series also
carries a distinct marker and dash pattern, so identity never rests on colour
alone -- required for greyscale print.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- geometry ------------------------------------------------------------
# \the\textwidth of sn-jnl.cls (sn-mathphys-num) = 372.0pt = 5.1467 in
TEXTWIDTH_IN = 372.0 / 72.27
def w(frac=1.0):
    return TEXTWIDTH_IN * frac

# --- palette (fixed order, never cycled) ---------------------------------
MODEL_ORDER = ["CP-MAE", "MSHTrans", "MTGFlow"]
MODEL_COLOR = {"CP-MAE": "#0072B2", "MSHTrans": "#009E73", "MTGFlow": "#D55E00"}
MODEL_MARKER = {"CP-MAE": "o", "MSHTrans": "s", "MTGFlow": "^"}
MODEL_DASH = {"CP-MAE": (None, None), "MSHTrans": (4, 1.6), "MTGFlow": (1.4, 1.4)}

COND_ORDER = ["clean", "c0.40"]
COND_LABEL = {"clean": r"clean ($\rho=0$)", "c0.40": r"contaminated ($\rho=0.4$)"}
COND_COLOR = {"clean": "#0072B2", "c0.40": "#D55E00"}
COND_MARKER = {"clean": "o", "c0.40": "^"}

DATASETS = ["SMD", "SWaT", "LTDB", "WADI", "PSM"]
# 5-slot categorical set, validated (worst adjacent CVD dE 11.0 deutan,
# normal-vision floor 25.8, all slots >= 3:1 contrast).  First three slots are
# identical to MODEL_COLOR so the two figures read as one system.
DS_COLOR = dict(zip(DATASETS, ["#0072B2", "#D55E00", "#009E73", "#9B4DCA", "#8A6E00"]))
DS_MARKER = dict(zip(DATASETS, ["o", "^", "s", "D", "v"]))
DS_DASH = dict(zip(DATASETS, [(None, None), (1.4, 1.4), (4, 1.6), (5, 1.5, 1.2, 1.5), (2.6, 1.2)]))

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
REF = "#9a9a9a"

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))   # CP-MAE_Revision/
RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))


def apply():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType, Springer requirement
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.6,
        "legend.frameon": False,
        "text.color": INK,
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, stem):
    """Write <stem>.pdf and <stem>.png next to main.tex, and report."""
    outs = []
    for ext in ("pdf", "png"):
        p = os.path.join(REPO, "%s.%s" % (stem, ext))
        fig.savefig(p)
        outs.append(p)
    plt.close(fig)
    for p in outs:
        print("  wrote %s (%.0f kB)" % (os.path.relpath(p, REPO),
                                        os.path.getsize(p) / 1024))
