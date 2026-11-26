"""Generate minimalist Keynote-style vector figures for `docs/research_overview.tex` (v0.4.1)."""

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
        "font.size": 10.6,
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
    """Minimalist scatter: 0.10B Val CE vs. Long-Range Copy Recall (>=64 tok, n=4,137)."""
    fig, ax = plt.subplots(figsize=(9.6, 4.25), dpi=300)

    # (label, copy_pct, val_ce, color, marker, size, (tx, ty), ha)
    points = [
        ("3:1 Hybrid Clean (B)  [KV -75%]", 80.64, 3.7123, TEAL, "*", 240, (66.5, 3.47), "right"),
        ("Transformer Clean (A)", 79.21, 3.7693, INK, "o", 88, (66.5, 3.72), "right"),
        ("3:1 94% W4 INT4+CIM6 (C)", 72.54, 3.9004, TEAL, "s", 96, (58.0, 3.89), "right"),
        ("3:1 94% 1.58b+CIM6", 68.31, 3.9765, GREEN, "D", 90, (54.5, 4.08), "right"),
        ("TF 94% W4 INT4+CIM6", 68.84, 4.0676, INK, "^", 86, (72.0, 4.18), "left"),
        ("TF 94% 1.58b+CIM6", 62.29, 4.0927, AMBER, "v", 86, (65.0, 4.46), "left"),
        ("Attention-Free Res (Clean)", 44.09, 4.2686, MUTED, "o", 78, (27.0, 4.55), "left"),
        ("3:1 SSM+Res 94% INT4+CIM6 (collapsed)", 36.48, 4.8866, CRIMSON, "P", 105, (43.5, 4.92), "left"),
        ("DeltaNet SSM HAT (diverged)", 33.41, 5.1592, CRIMSON, "X", 108, (43.5, 5.24), "left"),
    ]

    ax.text(87.2, 3.39, "Better  (higher copy, lower CE)", ha="right", va="top", fontsize=9.3, color=TEAL)

    for label, x_val, y_val, color, marker, size, (tx, ty), align in points:
        ax.scatter(x_val, y_val, c=color, marker=marker, s=size, edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(tx, ty),
            textcoords="data",
            ha=align,
            va="center",
            fontsize=9.2,
            fontweight="semibold" if color in (TEAL, GREEN, CRIMSON) else "normal",
            color=color,
            arrowprops={"arrowstyle": "-", "color": "#D1D5DB", "lw": 0.9, "shrinkA": 3, "shrinkB": 5},
            zorder=6,
        )

    ax.set_xlim(25.5, 88.5)
    ax.set_ylim(5.46, 3.30)
    ax.set_xlabel("Long-Range Copy Recall  >= 64 tokens  (%, n = 4,137)", fontsize=10.6, labelpad=7)
    ax.set_ylabel("Validation CE  (nats, lower = better)", fontsize=10.6, labelpad=7)
    ax.tick_params(labelsize=9.6)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fig2_phase2_factorized(out_path: Path) -> None:
    """Phase-2 equal-scope factorized ablation: Transformer vs 3:1 Spectral-Reservoir vs 3:1 SSM+Res (0.10B)."""
    fig, ax = plt.subplots(figsize=(9.9, 4.15), dpi=300)

    labels = [
        "Clean\nBF16",
        "94% Equal\nW4 INT4+CIM6",
        "100% W4\nINT4+CIM6",
        "94% Equal\n1.58b+CIM6",
        "100%\n1.58b Only",
        "3:1 SSM+Res\nClean vs CIM6",
    ]
    x = np.arange(len(labels))
    w = 0.34

    # First 5 groups: Transformer (Ink) vs 3:1 Spectral-Reservoir (Teal)
    tf_vals = [3.7693, 4.0676, 4.0871, 4.0927, 4.0978]
    res_vals = [3.7123, 3.9004, 3.9692, 3.9765, 3.8862]

    for i in range(5):
        ax.bar(x[i] - w / 2, tf_vals[i], width=w * 0.88, color=INK, zorder=3)
        ax.bar(x[i] + w / 2, res_vals[i], width=w * 0.88, color=TEAL, zorder=3)
        ax.text(x[i] - w / 2, tf_vals[i] + 0.035, f"{tf_vals[i]:.2f}", ha="center", va="bottom", fontsize=8.7, color=INK)
        ax.text(
            x[i] + w / 2,
            res_vals[i] + 0.035,
            f"{res_vals[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="semibold",
            color=TEAL,
        )
        delta = res_vals[i] - tf_vals[i]
        ax.text(
            x[i],
            max(tf_vals[i], res_vals[i]) + 0.22,
            f"({delta:+.2f})",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="semibold",
            color=GREEN,
        )

    # 6th group: 3:1 SSM+Res Clean (3.7829) vs 94% W4 INT4+CIM6 (4.8866, collapsed)
    ax.axvline(4.5, color="#D1D5DB", linestyle=":", linewidth=1.2)
    ax.bar(x[5] - w / 2, 3.7829, width=w * 0.88, color=MUTED, zorder=3)
    ax.bar(x[5] + w / 2, 4.8866, width=w * 0.88, color=CRIMSON, zorder=3)
    ax.text(x[5] - w / 2, 3.7829 + 0.035, "3.78", ha="center", va="bottom", fontsize=8.7, color=MUTED)
    ax.text(x[5] + w / 2, 4.8866 + 0.035, "4.89", ha="center", va="bottom", fontsize=9.0, fontweight="semibold", color=CRIMSON)

    ax.annotate(
        "Recurrent SSM collapses\n(+1.10 nats under 94% CIM6)",
        xy=(x[5] + w / 2 - 0.06, 4.86),
        xytext=(3.72, 4.92),
        ha="center",
        fontsize=8.9,
        fontweight="semibold",
        color=CRIMSON,
        arrowprops={"arrowstyle": "->", "color": CRIMSON, "lw": 1.1},
    )

    handles = [
        Patch(facecolor=INK, label="Transformer (A)  [KV 100%]"),
        Patch(facecolor=TEAL, label="3:1 Spectral-Reservoir  [KV 25%]"),
        Patch(facecolor=MUTED, label="3:1 SSM+Res (Clean)"),
        Patch(facecolor=CRIMSON, label="3:1 SSM+Res (94% INT4+CIM6)"),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.1)
    ax.set_xlim(-0.58, 5.60)
    ax.set_ylim(3.35, 5.45)
    ax.set_ylabel("Validation CE  (nats, lower = better)", fontsize=10.6, labelpad=7)
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=8.9)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fig3_systems_speed(out_path: Path) -> None:
    """Two-panel throughput plot: JAX v0.3/v0.4 & PyTorch A6000 BF16."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.85), dpi=300)

    # Left: JAX v0.2 -> v0.3/v0.4
    cats_j = ["L=512 (DP)\nv0.2", "L=512 (DP)\nv0.3/v0.4", "L=2048 (FSDP)\nv0.3/v0.4"]
    xj = np.arange(len(cats_j))
    w = 0.27

    ja = [7.321, 7.404, 8.981]
    jb = [5.054, 6.252, 7.365]
    jc = [4.772, 6.039, 6.593]

    for off, vals, base, color, show_r in (
        (-w, ja, ja, INK, False),
        (0.0, jb, ja, TEAL, True),
        (w, jc, ja, GREEN, True),
    ):
        ax1.bar(xj + off, vals, width=w * 0.86, color=color, zorder=3)
        for xi, v, bv in zip(xj, vals, base):
            ax1.text(xi + off, v + 0.20, f"{v:.1f}K", ha="center", va="bottom", fontsize=8.3, color=color, fontweight="semibold")
            if show_r:
                ax1.text(
                    xi + off,
                    v * 0.50,
                    f"{v / bv:.2f}x",
                    ha="center",
                    va="center",
                    fontsize=7.9,
                    color="white",
                    fontweight="semibold",
                    zorder=5,
                )

    ax1.set_xticks(xj)
    ax1.set_xticklabels(cats_j, fontsize=9.1)
    ax1.set_ylim(0, 10.6)
    ax1.set_ylabel("Valid Targets / sec  (K tok/s)", fontsize=10.1)
    ax1.set_title("JAX / XLA  (2x RTX 2080 Ti, FP32, Trace=1)", fontsize=10.1, pad=9, color=INK)
    ax1.legend(
        handles=[Patch(facecolor=INK, label="TF (A)"), Patch(facecolor=TEAL, label="Hybrid (B)"), Patch(facecolor=GREEN, label="W4 HAT (C)")],
        loc="upper left",
        frameon=False,
        fontsize=8.8,
    )

    # Right: PyTorch 2x A6000 BF16
    cats_p = ["0.10B\nClean (B)", "0.10B\nHAT (C)", "0.26B ckpt\nClean (B)", "0.26B ckpt\nHAT (C)"]
    xp = np.arange(len(cats_p))
    wp = 0.35

    before = [47.14, 31.68, 11.72, 7.17]
    after = [51.20, 50.96, 25.07, 19.13]
    gains = ["1.09x", "1.61x", "2.14x", "2.67x"]

    ax2.hlines(32.2, 1.55, 3.45, colors=INK, linestyles="--", linewidths=0.85, zorder=2)
    ax2.text(2.50, 34.0, "0.26B TF: 25.9-32.2K", ha="center", va="bottom", fontsize=8.4, color=INK, zorder=5)

    ax2.hlines(77.4, -0.45, 1.45, colors=INK, linestyles="--", linewidths=0.85, zorder=2)
    ax2.text(0.50, 79.2, "0.10B TF: ~69-77K", ha="center", va="bottom", fontsize=8.4, color=INK, zorder=5)

    ax2.bar(xp - wp / 2, before, width=wp * 0.90, color="#D1D5DB", zorder=3)
    colors_after = [TEAL, GREEN, TEAL, GREEN]
    ax2.bar(xp + wp / 2, after, width=wp * 0.90, color=colors_after, zorder=3)

    for xi, v in zip(xp, before):
        ax2.text(xi - wp / 2, v + 1.3, f"{v:.1f}K", ha="center", va="bottom", fontsize=8.3, color=MUTED, zorder=6)
    for xi, v, g, col in zip(xp, after, gains, colors_after):
        ax2.text(xi + wp / 2, v + 1.3, f"{v:.1f}K", ha="center", va="bottom", fontsize=8.5, color=col, fontweight="semibold", zorder=6)
        ax2.text(xi + wp / 2, v * 0.50, g, ha="center", va="center", fontsize=8.2, color="white", fontweight="semibold", zorder=6)

    ax2.set_xticks(xp)
    ax2.set_xticklabels(cats_p, fontsize=9.0)
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Training Throughput  (K tok/s)", fontsize=10.1)
    ax2.set_title("PyTorch BF16  (2x RTX A6000, FlexAttn, L=2048)", fontsize=10.1, pad=9, color=INK)
    ax2.legend(
        handles=[
            Patch(facecolor="#D1D5DB", label="Before"),
            Patch(facecolor=TEAL, label="After (Clean B)"),
            Patch(facecolor=GREEN, label="After (HAT C)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.5,
    )

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fig4_real_audio(out_path: Path) -> None:
    """Two-panel causal 12.5 Hz Japanese audio turn-taking pilot (IngCrowd Sessions 09-10, n=196, 2-seed)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.7, 3.95), dpi=300)

    # Left: Floor Hold vs. Shift AUROC across Clean, SNR 10 dB, SNR 3 dB + Crosstalk (mean ± half-range)
    conds_a = ["Clean\nSpeech", "Field Noise\nSNR 10 dB", "Harsh + Crosstalk\nSNR 3 dB"]
    xa = np.arange(len(conds_a))
    wa = 0.20

    vad_auroc = [0.7255, 0.6672, 0.6208]
    txt_auroc = [0.6641, 0.6641, 0.6641]
    txt_err = [0.0083 / 2, 0.0083 / 2, 0.0083 / 2]
    mimi_auroc = [0.7256, 0.6816, 0.6694]
    mimi_err = [0.0110 / 2, 0.0091 / 2, 0.0508 / 2]
    fus_auroc = [0.7007, 0.6615, 0.6518]
    fus_err = [0.0227 / 2, 0.0224 / 2, 0.0177 / 2]

    ax1.bar(xa - 1.5 * wa, vad_auroc, width=wa * 0.86, color="#D1D5DB", zorder=3)
    ax1.bar(xa - 0.5 * wa, txt_auroc, width=wa * 0.86, yerr=txt_err, capsize=2.5, color=AMBER, zorder=3)
    ax1.bar(xa + 0.5 * wa, mimi_auroc, width=wa * 0.86, yerr=mimi_err, capsize=2.5, color=TEAL, zorder=3)
    ax1.bar(xa + 1.5 * wa, fus_auroc, width=wa * 0.86, yerr=fus_err, capsize=2.5, color=GREEN, zorder=3)

    for xi in range(3):
        ax1.text(
            xa[xi] + 0.53 * wa,
            mimi_auroc[xi] + mimi_err[xi] + 0.010,
            f"{mimi_auroc[xi]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=TEAL,
            fontweight="semibold",
        )
        ax1.text(
            xa[xi] - 0.53 * wa,
            txt_auroc[xi] + txt_err[xi] + 0.010,
            f"{txt_auroc[xi]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color=AMBER,
        )

    ax1.annotate(
        "+6.2 pt vs. 0%-WER text\n(0 ms wait, zero ASR)",
        xy=(0.5 * wa, 0.752),
        xytext=(0.56, 0.824),
        ha="center",
        fontsize=8.3,
        fontweight="semibold",
        color=TEAL,
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.0},
    )

    ax1.set_xticks(xa)
    ax1.set_xticklabels(conds_a, fontsize=9.1)
    ax1.set_ylim(0.50, 0.88)
    ax1.set_ylabel("Floor Hold vs. Shift AUROC  (higher = better)", fontsize=9.7)
    ax1.set_title("0-ms Floor-Shift AUROC  (IngCrowd, n = 196, 2-seed)", fontsize=9.9, pad=9, color=INK)
    ax1.legend(
        handles=[
            Patch(facecolor="#D1D5DB", label="Energy VAD (+800 ms)"),
            Patch(facecolor=AMBER, label="0%-WER Text (0 ms)"),
            Patch(facecolor=TEAL, label="Mimi 12.5 Hz (0 ASR)"),
            Patch(facecolor=GREEN, label="Mimi+Text (3:1 W4+CIM6)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=7.9,
    )

    # Right: Long-Pause Premature Takeover Rate (%, lower = better)
    conds_b = ["Clean Pause\n>= 500 ms (n=57)", "Clean Pause\n>= 800 ms (n=17)", "SNR 3 dB + XT\n>= 800 ms (n=17)"]
    xb = np.arange(len(conds_b))
    wb = 0.20

    vad_pre = [42.11, 88.24, 82.35]
    txt_pre = [55.26, 52.94, 52.94]
    txt_pe = [5.26 / 2, 0.0, 0.0]
    mimi_pre = [36.84, 50.00, 64.71]
    mimi_pe = [0.0, 5.88 / 2, 0.0]
    fus_pre = [45.62, 47.06, 41.18]
    fus_pe = [10.53 / 2, 11.76 / 2, 11.77 / 2]

    ax2.bar(xb - 1.5 * wb, vad_pre, width=wb * 0.86, color="#D1D5DB", zorder=3)
    ax2.bar(xb - 0.5 * wb, txt_pre, width=wb * 0.86, yerr=txt_pe, capsize=2.5, color=AMBER, zorder=3)
    ax2.bar(xb + 0.5 * wb, mimi_pre, width=wb * 0.86, yerr=mimi_pe, capsize=2.5, color=TEAL, zorder=3)
    ax2.bar(xb + 1.5 * wb, fus_pre, width=wb * 0.86, yerr=fus_pe, capsize=2.5, color=GREEN, zorder=3)

    for xi in range(3):
        ax2.text(
            xb[xi] - 1.5 * wb,
            vad_pre[xi] + 2.2,
            f"{vad_pre[xi]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=MUTED,
        )
        ax2.text(
            xb[xi] + 0.48 * wb,
            mimi_pre[xi] + mimi_pe[xi] + 2.4,
            f"{mimi_pre[xi]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=TEAL,
            fontweight="semibold",
        )
        ax2.text(
            xb[xi] + 1.55 * wb,
            fus_pre[xi] + fus_pe[xi] + 2.4,
            f"{fus_pre[xi]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=GREEN,
            fontweight="semibold",
        )

    ax2.annotate(
        "-18.4 pt vs. text\n(zero ASR)",
        xy=(0.5 * wb, 43.5),
        xytext=(0.18, 76.5),
        ha="center",
        fontsize=8.3,
        fontweight="semibold",
        color=TEAL,
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.0},
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels(conds_b, fontsize=9.0)
    ax2.set_ylim(0.0, 106.0)
    ax2.set_ylabel("Premature Takeover Rate  (%, lower = better)", fontsize=9.7)
    ax2.set_title("Long-Pause False Interruptions  (2-seed mean ± spread)", fontsize=9.9, pad=9, color=INK)

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "docs" / "figs"
    for old in (
        "fig1_pilot_tradeoff.pdf",
        "fig2_systems_throughput.pdf",
        "fig2_cim_resilience.pdf",
        "fig4_downstream_s2s_audio.pdf",
    ):
        p = fig_dir / old
        if p.exists():
            p.unlink()
    build_fig1_copy_vs_ce(fig_dir / "fig1_copy_vs_ce.pdf")
    build_fig2_phase2_factorized(fig_dir / "fig2_phase2_factorized.pdf")
    build_fig3_systems_speed(fig_dir / "fig3_systems_speed.pdf")
    build_fig4_real_audio(fig_dir / "fig4_real_audio.pdf")
    print(f"Generated v0.4.1 minimalist Keynote figures under {fig_dir}")


if __name__ == "__main__":
    main()
