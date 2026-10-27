"""Shared matplotlib styling + figure saving (300 dpi PNGs to reports/figures/).

Colours follow a validated, colour-blind-safe categorical order: slot 1 (blue)
= no failure, slot 2 (orange) = failure. Correlations use a blue<->red diverging
map with a neutral grey midpoint.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIG_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_OK, C_FAIL = SERIES[0], SERIES[1]
CLASS_COLORS = {0: C_OK, 1: C_FAIL}
CLASS_LABELS = {0: "No failure", 1: "Failure"}

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32424"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def savefig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")


# ------------------------------------------------------- evaluation plots
def plot_pr_curves(ax, y, curves: dict, prevalence: float | None = None):
    """curves: {label: oof_proba}. Draws PR curves with AP in the legend."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    for i, (label, p) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.step(rec, prec, where="post", color=SERIES[i], lw=2,
                label=f"{label}  (AP = {average_precision_score(y, p):.3f})")
    if prevalence is not None:
        ax.axhline(prevalence, color=MUTED, lw=1, ls="--", label=f"Chance = prevalence ({prevalence:.1%})")
    ax.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="Recall (failures caught)",
           ylabel="Precision (alarms that are real)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.05), fontsize=9)


def plot_roc_curves(ax, y, curves: dict):
    from sklearn.metrics import roc_auc_score, roc_curve

    for i, (label, p) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=SERIES[i], lw=2, label=f"{label}  (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate (recall)")
    ax.legend(loc="lower right")


def plot_confusion(ax, cm, title: str = ""):
    """cm = [[tn, fp], [fn, tp]] with row-normalised shading and raw counts printed."""
    import numpy as np

    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    names = ["No failure", "Failure"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%} of row", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], names); ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)
