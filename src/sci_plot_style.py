"""Shared publication-quality Matplotlib settings for the wind-reconstruction paper."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Colour-blind-safe, print-friendly palette (Okabe--Ito inspired).
PALETTE = {
    "ohats": "#000000",
    "stage3": "#0072B2",
    "proposed": "#D55E00",
    "proposed_no_wave": "#009E73",
    "background": "#7F7F7F",
    "unstable": "#D55E00",
    "near_neutral": "#0072B2",
    "stable": "#009E73",
    "fill": "#B8B8B8",
}

LINE_STYLES = {
    "ohats": "-",
    "stage3": "--",
    "proposed": "-",
    "proposed_no_wave": "-.",
}

MARKERS = {
    "ohats": "o",
    "stage3": "s",
    "proposed": "^",
    "proposed_no_wave": "D",
}


def _select_serif_font() -> str:
    candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    available = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            if candidate != "Times New Roman":
                warnings.warn(
                    f"Times New Roman was not found. Falling back to {candidate}. "
                    "On Windows, install/use the system Times New Roman font before final export.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return candidate
    return "DejaVu Serif"


def apply_sci_style(base_font_size: float = 8.5) -> str:
    """Apply a compact SCI-journal figure style and return the selected font name."""
    selected_font = _select_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [selected_font, "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "axes.titlesize": base_font_size,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "xtick.labelsize": base_font_size - 0.5,
            "ytick.labelsize": base_font_size - 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": base_font_size - 0.8,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.22,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "figure.dpi": 120,
        }
    )
    return selected_font


def panel_label(ax: plt.Axes, label: str, x: float = 0.01, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=9.0,
    )


def clean_axis(ax: plt.Axes, *, grid: bool = True, log_grid: bool = False) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if grid:
        ax.grid(True, which="both" if log_grid else "major")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_figure(fig: plt.Figure, output_base: Path, *, close: bool = True) -> list[Path]:
    """Save both 600 dpi PNG and vector PDF versions."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    if close:
        plt.close(fig)
    return [png_path, pdf_path]
