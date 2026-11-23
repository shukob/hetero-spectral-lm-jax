"""Generate minimalist Keynote-style vector figures for `docs/research_overview.tex`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
TEAL = "#0F766E"
GREEN = "#15803D"
CRIMSON = "#B91C1C"
AMBER = "#B45309"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def build_fig1_copy_vs_ce(out_path: Path) -> None:
    """Single clean minimalist scatter: 0.10B Val CE vs. Long-Range Copy Recall (>=64 tok)."""
    fig, ax = plt.subplots(figsize=(9.4, 4.15), dpi=300)

    points = [
        ("3:1 Hybrid Clean (B)  [KV -75%]", 80.95, 3.7124, TEAL, "*", 250, (44.0, 3.55)),
        ("Transformer Clean (A)", 79.21, 3.7693, INK, "o", 90, (49.0, 3.81)),
        ("93.8% Analog HAT (C)", 68.31, 3.9765, GREEN, "D", 95, (69.5, 4.35)),
        ("3:1 Hybrid 100% HAT", 56.56, 4.0821, "#16A34A", "s", 75, (60.5, 4.74)),
        ("Transformer 100% HAT", 46.34, 4.2594, AMBER, "^", 85, (30.5, 4.01)),
        ("Attention-Free Res (Clean)", 44.09, 4.2686, MUTED, "v", 80, (30.5, 4.66)),
        ("DeltaNet SSM HAT  (diverged)", 33.41, 5.1592, CRIMSON, "X", 115, (38.0, 5.18)),
    ]

    ax.axvspan(65.0, 86.0, ymin=0.58, ymax=1.0, color="#F0FDFA", alpha=0.85, zorder=0)
    ax.text(85.0, 3.43, "Better  (higher copy, lower CE)", ha="right", va="top", fontsize=9.6, color=TEAL)

    for label, x_val, y_val, color, marker, size, (tx, ty) in points:
        ax.scatter(x_val, y_val, c=color, marker=marker, s=size, edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=10.0,
            fontweight="semibold" if color in (TEAL, GREEN, CRIMSON) else "normal",
            color=color,
            arrowprops={"arrowstyle": "-", "color": "#D1D5DB", "lw": 0.9, "shrinkA": 3, "shrinkB": 5},
            zorder=6,
        )

    ax.set_xlim(28, 86)
    ax.set_ylim(5.40, 3.36)
    ax.set_xlabel("Long-Range Copy Recall  >= 64 tokens  (%, n = 4,137)", fontsize=11.0, labelpad=7)
    ax.set_ylabel("Validation CE  (nats, lower = better)", fontsize=11.0, labelpad=7)
    ax.tick_params(labelsize=10.0)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fig2_cim_resilience(out_path: Path) -> None:
    """Single clean minimalist bar chart: Scale trend & 1.58-bit + 6-bit Analog CIM eval."""
    fig, ax = plt.subplots(figsize=(9.4, 4.15), dpi=300)

    labels = [
        "0.10B\nClean",
        "0.26B\nClean",
        "0.10B\n6b CIM Active",
        "0.26B\n6b CIM Active",
        "0.10B Deploy\n1.58b + 6b CIM",
        "0.26B Deploy\n1.58b + 6b CIM",
    ]
    x = np.arange(len(labels))
    w = 0.25

    vals_a = [3.769, 3.724, 4.259, np.nan, 6.369, np.nan]
    vals_b = [3.712, 3.692, np.nan, np.nan, 5.379, 5.492]
    vals_c = [np.nan, np.nan, 3.977, 3.899, 4.093, 4.016]

    for offset, vals, color in ((-w, vals_a, INK), (0.0, vals_b, TEAL), (w, vals_c, GREEN)):
        for xi, v in zip(x, vals):
            if np.isnan(v):
                continue
            ax.bar(xi + offset, v, width=w * 0.90, color=color, zorder=3)
            ax.text(
                xi + offset,
                v + 0.08,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="semibold",
                color=color,
            )

    ax.axvline(1.5, color="#D1D5DB", linestyle=":", linewidth=1.1)
    ax.axvline(3.5, color="#D1D5DB", linestyle=":", linewidth=1.1)

    ax.annotate(
        "Gap vs. clean B narrows\n+0.264  ->  +0.206 nats",
        xy=(3 + w, 4.26),
        xytext=(2.35, 5.48),
        ha="center",
        fontsize=9.6,
        color=GREEN,
        fontweight="semibold",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.1},
    )
    ax.annotate(
        "4-bit harsh: 4.06\nchip spread <= 0.002",
        xy=(5 + w, 4.35),
        xytext=(5 + w, 6.18),
        ha="center",
        fontsize=9.3,
        color=GREEN,
        fontweight="semibold",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.1},
    )

    handles = [
        Patch(facecolor=INK, label="Model A: Transformer"),
        Patch(facecolor=TEAL, label="Model B: 3:1 Hybrid (Clean)"),
        Patch(facecolor=GREEN, label="Model C: 94% Analog HAT"),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.6)
    ax.set_xlim(-0.55, 5.85)
    ax.set_ylim(2.8, 6.95)
    ax.set_ylabel("Validation CE  (nats, lower = better)", fontsize=11.0, labelpad=7)
    ax.tick_params(axis="y", labelsize=10.0)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.8)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fig3_systems_speed(out_path: Path) -> None:
    """Clean minimalist two-panel throughput plot: JAX v0.3 & PyTorch A6000 BF16."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.85), dpi=300)

    # Left: JAX v0.2 -> v0.3
    cats_j = ["L=512 (DP)\nv0.2", "L=512 (DP)\nv0.3", "L=2048 (FSDP)\nv0.3"]
    xj = np.arange(len(cats_j))
    w = 0.27

    ja = [7.321, 7.404, 8.981]
    jb = [5.054, 6.252, 7.365]
    jc = [4.772, 6.017, 7.198]

    for off, vals, base, color, show_r in (
        (-w, ja, ja, INK, False),
        (0.0, jb, ja, TEAL, True),
        (w, jc, ja, GREEN, True),
    ):
        ax1.bar(xj + off, vals, width=w * 0.86, color=color, zorder=3)
        for xi, v, bv in zip(xj, vals, base):
            ax1.text(xi + off, v + 0.20, f"{v:.1f}K", ha="center", va="bottom", fontsize=8.4, color=color, fontweight="semibold")
            if show_r:
                ax1.text(
                    xi + off,
                    v * 0.50,
                    f"{v / bv:.2f}x",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="white",
                    fontweight="semibold",
                    zorder=5,
                )

    ax1.set_xticks(xj)
    ax1.set_xticklabels(cats_j, fontsize=9.4)
    ax1.set_ylim(0, 10.6)
    ax1.set_ylabel("Valid Targets / sec  (K tok/s)", fontsize=10.2)
    ax1.set_title("JAX / XLA  (2x RTX 2080 Ti, FP32, Trace=1)", fontsize=10.4, pad=9, color=INK)
    ax1.legend(
        handles=[Patch(facecolor=INK, label="TF (A)"), Patch(facecolor=TEAL, label="Hybrid (B)"), Patch(facecolor=GREEN, label="HAT (C)")],
        loc="upper left",
        frameon=False,
        fontsize=9.0,
    )

    # Right: PyTorch 2x A6000 BF16
    cats_p = ["0.10B\nClean (B)", "0.10B\nHAT (C)", "0.26B ckpt\nClean (B)", "0.26B ckpt\nHAT (C)"]
    xp = np.arange(len(cats_p))
    wp = 0.35

    before = [47.14, 31.68, 11.72, 7.17]
    after = [51.20, 49.70, 25.07, 19.13]
    gains = ["1.09x", "1.57x", "2.14x", "2.67x"]

    # Subtle shaded TF reference ranges behind bars (zorder=1)
    ax2.fill_between([1.55, 3.45], 25.9, 32.2, color="#F3F4F6", zorder=1)
    ax2.hlines(32.2, 1.55, 3.45, colors=INK, linestyles="--", linewidths=0.85, zorder=2)
    ax2.text(2.50, 34.0, "0.26B TF: 25.9-32.2K", ha="center", va="bottom", fontsize=8.4, color=INK, zorder=5)

    ax2.hlines(77.4, -0.45, 1.45, colors=INK, linestyles="--", linewidths=0.85, zorder=2)
    ax2.text(0.50, 79.2, "0.10B TF: ~70-77K", ha="center", va="bottom", fontsize=8.4, color=INK, zorder=5)

    ax2.bar(xp - wp / 2, before, width=wp * 0.90, color="#D1D5DB", zorder=3)
    colors_after = [TEAL, GREEN, TEAL, GREEN]
    ax2.bar(xp + wp / 2, after, width=wp * 0.90, color=colors_after, zorder=3)

    for xi, v in zip(xp, before):
        ax2.text(xi - wp / 2, v + 1.3, f"{v:.1f}K", ha="center", va="bottom", fontsize=8.3, color=MUTED, zorder=6)
    for xi, v, g, col in zip(xp, after, gains, colors_after):
        ax2.text(xi + wp / 2, v + 1.3, f"{v:.1f}K", ha="center", va="bottom", fontsize=8.5, color=col, fontweight="semibold", zorder=6)
        ax2.text(
            xi + wp / 2,
            v * 0.50,
            g,
            ha="center",
            va="center",
            fontsize=8.3,
            color="white",
            fontweight="semibold",
            zorder=6,
        )

    ax2.set_xticks(xp)
    ax2.set_xticklabels(cats_p, fontsize=9.0)
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Training Throughput  (K tok/s)", fontsize=10.2)
    ax2.set_title("PyTorch BF16  (2x RTX A6000, FlexAttn, L=2048)", fontsize=10.4, pad=9, color=INK)
    ax2.legend(
        handles=[
            Patch(facecolor="#D1D5DB", label="Before"),
            Patch(facecolor=TEAL, label="After (Clean B)"),
            Patch(facecolor=GREEN, label="After (HAT C)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.6,
    )

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "docs" / "figs"
    for old in ("fig1_pilot_tradeoff.pdf", "fig2_systems_throughput.pdf"):
        p = fig_dir / old
        if p.exists():
            p.unlink()
    build_fig1_copy_vs_ce(fig_dir / "fig1_copy_vs_ce.pdf")
    build_fig2_cim_resilience(fig_dir / "fig2_cim_resilience.pdf")
    build_fig3_systems_speed(fig_dir / "fig3_systems_speed.pdf")
    print(f"Generated minimalist Keynote figures under {fig_dir}")


if __name__ == "__main__":
    main()
