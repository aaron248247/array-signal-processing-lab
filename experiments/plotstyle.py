"""Shared matplotlib style for the experiment figures (plotting only; no computation)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from array_signal_lab.io import FIGURES_DIR  # noqa: E402

# Fixed categorical order (validated colour-blind-separable set); line styles add a
# second encoding so no curve is identified by colour alone.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax, letter: str) -> None:
    ax.set_title(f"({letter})", loc="left", fontweight="bold")


def save(fig, filename: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)
