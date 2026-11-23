"""Generate clean vector figures for `docs/research_overview.tex` from verified pilot & v0.3.0 benchmark measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

NAVY = "#0F2942"
TEAL = "#007A87"
GREEN = "#2F855A"
CORAL = "#C53030"
AMBER = "#B7791F"
SLATE = "#4A5568"
LIGHT_GRID = "#E2E8F0"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.edgecolor": "#CBD5E0",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": LIGHT_GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)


def build_fig1_pilot_tradeoff(out_path: Path) -> None:
    """Two-panel pilot figure: (L) Copy Recall vs Val CE at 0.10B, (R) Scale & 6b/4b CIM Resilience."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.9, 3.65), dpi=300, gridspec_kw={"width_ratios": [1.06, 1.0]})

    # --- Left Panel: 0.10B Copy Recall (>=64 tok, n=4,137) vs Val CE (26.2M tok, 2xA6000 BF16) ---
    # Explicit data-coordinate label anchors guarantee zero overlap across all 7 markers.
    points = [
        ("Model B: 3:1 Res (Clean, KV -75%)", 80.95, 3.7124, TEAL, "*", 175, (40.0, 3.55)),
        ("Model A: Transformer (Clean)", 79.21, 3.7693, NAVY, "o", 70, (40.0, 3.78)),
        ("Model C: 93.8% Analog HAT\n(6% Digital GQA)", 68.31, 3.9765, GREEN, "D", 76, (66.5, 4.44)),
        ("3:1 Res 100% Analog HAT", 56.56, 4.0821, "#2F855A", "s", 58, (58.0, 4.82)),
        ("Transformer 100% HAT", 46.34, 4.2594, AMBER, "^", 66, (29.5, 4.04)),
        ("Attention-Free Res (Clean)", 44.09, 4.2686, SLATE, "v", 62, (29.5, 4.66)),
        ("DeltaNet SSM HAT\n(Diverged)", 33.41, 5.1592, CORAL, "X", 88, (37.5, 5.18)),
    ]

    ax1.axvspan(65.0, 86.5, ymin=0.60, ymax=1.0, color="#E6FFFA", alpha=0.45, zorder=0)
    ax1.text(85.5, 3.51, "Optimal Zone", ha="right", va="top", fontsize=7.4, color=TEAL, fontweight="bold")

    for label, copy_pct, ce, color, marker, size, (tx, ty) in points:
        ax1.scatter(copy_pct, ce, c=color, marker=marker, s=size, edgecolors="white", linewidths=0.9, zorder=5)
        ax1.annotate(
            label,
            xy=(copy_pct, ce),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=7.0,
            fontweight="bold" if color in (TEAL, GREEN, CORAL) else "normal",
            color=color,
            arrowprops={"arrowstyle": "-", "color": "#A0AEC0", "lw": 0.75, "shrinkA": 2, "shrinkB": 4},
            zorder=6,
        )

    ax1.set_xlim(28, 86.5)
    ax1.set_ylim(5.42, 3.45)  # inverted: lower CE is higher on plot
    ax1.set_xlabel("Long-Range Copy Recall >= 64 tok (%, n=4,137)", fontweight="bold", color=NAVY)
    ax1.set_ylabel("Validation Loss (nats, lower = better)", fontweight="bold", color=NAVY)
    ax1.set_title("(A) 0.10B Pilot: Accuracy vs. Copy Recall (26.2M tok, 1 seed)", fontsize=9.2, fontweight="bold", color=NAVY, pad=6)

    # --- Right Panel: Scale Signal & Analog CIM Resilience ---
    labels = [
        "0.10B\nClean",
        "0.26B\nClean",
        "0.10B\n6b CIM\n(Active)",
        "0.26B\n6b CIM\n(Active)",
        "Deploy\n1.58b+6b\n(0.10B)",
        "Deploy\n1.58b+6b\n(0.26B)",
    ]
    x = np.arange(len(labels))
    w = 0.27

    vals_a = [3.769, 3.724, 4.259, np.nan, 6.369, np.nan]
    vals_b = [3.712, 3.692, np.nan, np.nan, 5.379, 5.492]
    vals_c = [np.nan, np.nan, 3.977, 3.899, 4.093, 4.016]

    for offset, vals, color in ((-w, vals_a, NAVY), (0.0, vals_b, TEAL), (w, vals_c, GREEN)):
        for xi, v in zip(x, vals):
            if np.isnan(v):
                continue
            ax2.bar(xi + offset, v, width=w * 0.94, color=color, zorder=3)
            ax2.text(
                xi + offset,
                v + 0.06,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color=NAVY,
            )

    ax2.axvline(1.5, color="#CBD5E0", linestyle=":", linewidth=1.1)
    ax2.axvline(3.5, color="#CBD5E0", linestyle=":", linewidth=1.1)
    ax2.annotate(
        "Gap vs B narrows:\n+0.264 -> +0.206",
        xy=(3 + w, 4.18),
        xytext=(2.05, 5.55),
        fontsize=7.1,
        fontweight="bold",
        color=GREEN,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#F0FFF4", "ec": GREEN, "lw": 0.8},
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 0.9},
    )
    ax2.annotate(
        "4b harsh: 4.06\nchip d<=0.002",
        xy=(5 + w, 4.30),
        xytext=(4.45, 6.18),
        fontsize=6.9,
        fontweight="bold",
        color=GREEN,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#F0FFF4", "ec": GREEN, "lw": 0.7},
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 0.85},
    )

    handles = [
        Patch(facecolor=NAVY, label="Model A: Transformer"),
        Patch(facecolor=TEAL, label="Model B: 3:1 Res (Clean)"),
        Patch(facecolor=GREEN, label="Model C: 94% Analog HAT"),
    ]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.2)
    ax2.set_ylim(2.9, 6.95)
    ax2.set_ylabel("Validation CE (nats, lower = better)", fontweight="bold", color=NAVY)
    ax2.set_title("(B) Scale Trend & 1.58-bit + 6-bit Analog CIM Eval", fontsize=9.2, fontweight="bold", color=NAVY, pad=6)
    ax2.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=True, facecolor="white", framealpha=0.94)

    fig.tight_layout(pad=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_fig2_systems_throughput(out_path: Path) -> None:
    """Two-panel systems throughput figure: (L) JAX v0.2->v0.3 GPU smoke, (R) PyTorch 2xA6000 BF16 opt."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.9, 3.45), dpi=300)

    # --- Left Panel: JAX v0.2 vs v0.3 (2x RTX 2080 Ti FP32, 0.10B, 1 XLA trace) ---
    cats_j = ["L=512 (DP)\nv0.2 Baseline", "L=512 (DP)\nv0.3 Optimized", "L=2048 (FSDP)\nv0.3 Optimized"]
    xj = np.arange(len(cats_j))
    w = 0.25

    ja = [7320.9, 7404.1, 8980.7]
    jb = [5054.0, 6252.0, 7365.3]
    jc = [4771.6, 6017.4, 7197.9]

    ba = ax1.bar(xj - w, [v / 1000.0 for v in ja], width=w, color=NAVY, label="Model A: Transformer", zorder=3)
    bb = ax1.bar(xj, [v / 1000.0 for v in jb], width=w, color=TEAL, label="Model B: 3:1 Res", zorder=3)
    bc = ax1.bar(xj + w, [v / 1000.0 for v in jc], width=w, color=GREEN, label="Model C: 94% HAT", zorder=3)

    for bars, vals, base_vals, show_ratio in ((ba, ja, ja, False), (bb, jb, ja, True), (bc, jc, ja, True)):
        for bar, v, bv in zip(bars, vals, base_vals):
            txt = f"{v/1000:.2f}K\n({v/bv:.2f}x)" if show_ratio else f"{v/1000:.2f}K"
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                v / 1000.0 + 0.15,
                txt,
                ha="center",
                va="bottom",
                fontsize=6.7,
                fontweight="bold",
                color=NAVY,
            )

    ax1.set_xticks(xj)
    ax1.set_xticklabels(cats_j, fontsize=7.9)
    ax1.set_ylim(0, 10.9)
    ax1.set_ylabel("Steady Valid Targets / sec (K tok/s)", fontweight="bold", color=NAVY)
    ax1.set_title("(A) JAX/XLA v0.3.0 (2x RTX 2080 Ti FP32, 0.10B, Trace=1)", fontsize=9.1, fontweight="bold", color=NAVY, pad=6)
    ax1.legend(loc="upper left", fontsize=7.1, frameon=True, facecolor="white")

    # --- Right Panel: PyTorch BF16 Optimization on 2x RTX A6000 (FlexAttention, L=2048, real corpus) ---
    cats_p = ["0.10B\nModel B\n(Clean)", "0.10B\nModel C\n(94% HAT)", "0.26B ckpt\nModel B\n(Clean)", "0.26B ckpt\nModel C\n(94% HAT)"]
    xp = np.arange(len(cats_p))
    wp = 0.34

    before = [47.14, 31.68, 11.72, 7.17]
    after = [51.20, 49.70, 25.07, 19.13]
    gains = ["1.09x", "1.57x", "2.14x", "2.67x"]

    b0 = ax2.bar(xp - wp / 2, before, width=wp, color="#A0AEC0", zorder=3)
    b1 = ax2.bar(xp + wp / 2, after, width=wp, color=TEAL, zorder=3)
    b1[1].set_color(GREEN)
    b1[3].set_color(GREEN)

    for bar, v in zip(b0, before):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, v + 0.9, f"{v:.1f}K", ha="center", va="bottom", fontsize=6.7, color=SLATE)
    for idx_b, (bar, v, g) in enumerate(zip(b1, after, gains)):
        y_off = 0.6 if idx_b == 3 else (2.4 if idx_b == 2 else 1.0)
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + y_off,
            f"{v:.1f}K\n({g})",
            ha="center",
            va="bottom",
            fontsize=6.5 if idx_b >= 2 else 7.0,
            fontweight="bold",
            color=NAVY,
        )

    # Reference TF dashed lines with clean non-overlapping callouts
    ax2.hlines(77.4, -0.45, 1.45, colors=NAVY, linestyles="--", linewidths=1.0, zorder=4)
    ax2.text(0.5, 79.0, "0.10B TF Ref: ~70-77K tok/s", ha="center", va="bottom", fontsize=7.0, color=NAVY, fontweight="bold")
    ax2.hlines(25.9, 1.55, 3.45, colors=NAVY, linestyles="--", linewidths=1.0, zorder=4)
    ax2.annotate(
        "0.26B TF Ref: 25.9-32.2K",
        xy=(2.78, 26.2),
        xytext=(2.58, 46.5),
        ha="center",
        fontsize=7.0,
        fontweight="bold",
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#F7FAFC", "ec": NAVY, "lw": 0.7},
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 0.85},
    )

    handles_p = [
        Patch(facecolor="#A0AEC0", label="Before (Eager Res + Python Doc Loop)"),
        Patch(facecolor=TEAL, label="After: Clean B (Compile + Vec FFT + Toeplitz)"),
        Patch(facecolor=GREEN, label="After: HAT C (Compile + Vec FFT + Toeplitz)"),
    ]
    ax2.set_xticks(xp)
    ax2.set_xticklabels(cats_p, fontsize=7.6)
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Steady Training Throughput (K tok/s)", fontweight="bold", color=NAVY)
    ax2.set_title("(B) PyTorch BF16 Opt (2x RTX A6000, FlexAttn, L=2048)", fontsize=9.1, fontweight="bold", color=NAVY, pad=6)
    ax2.legend(handles=handles_p, loc="upper right", fontsize=6.5, frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout(pad=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "docs" / "figs"
    build_fig1_pilot_tradeoff(fig_dir / "fig1_pilot_tradeoff.pdf")
    build_fig2_systems_throughput(fig_dir / "fig2_systems_throughput.pdf")
    print(f"Generated figures under {fig_dir}")


if __name__ == "__main__":
    main()
