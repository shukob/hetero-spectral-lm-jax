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
        "font.size": 10.8,
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
    fig, ax = plt.subplots(figsize=(9.5, 4.25), dpi=300)

    points = [
        ("3:1 Hybrid Clean (B)  [KV -75%]", 80.64, 3.7123, TEAL, "*", 250, (37.5, 3.49)),
        ("Transformer Clean (A)", 79.21, 3.7693, INK, "o", 90, (42.5, 3.78)),
        ("3:1 94% 1.58b+CIM6", 68.31, 3.9765, GREEN, "D", 95, (69.5, 4.26)),
        ("3:1 100% W4 INT4+CIM6", 64.95, 3.9692, TEAL, "s", 88, (35.5, 4.07)),
        ("TF 94% 1.58b+CIM6 (equal-scope)", 62.30, 4.0927, AMBER, "^", 85, (62.5, 4.66)),
        ("Attention-Free Res (Clean)", 44.09, 4.2686, MUTED, "v", 80, (27.5, 4.50)),
        ("3:1 SSM+Res INT4+CIM6 (collapsed)", 36.50, 4.8866, CRIMSON, "P", 105, (45.0, 4.94)),
        ("DeltaNet SSM HAT (diverged)", 33.41, 5.1592, CRIMSON, "X", 110, (45.0, 5.27)),
    ]

    ax.axvspan(63.0, 87.0, ymin=0.60, ymax=1.0, color="#F0FDFA", alpha=0.85, zorder=0)
    ax.text(86.0, 3.39, "Better  (higher copy, lower CE)", ha="right", va="top", fontsize=9.5, color=TEAL)

    for label, x_val, y_val, color, marker, size, (tx, ty) in points:
        ax.scatter(x_val, y_val, c=color, marker=marker, s=size, edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=9.5,
            fontweight="semibold" if color in (TEAL, GREEN, CRIMSON) else "normal",
            color=color,
            arrowprops={"arrowstyle": "-", "color": "#D1D5DB", "lw": 0.9, "shrinkA": 3, "shrinkB": 5},
            zorder=6,
        )

    ax.set_xlim(26, 87)
    ax.set_ylim(5.45, 3.32)
    ax.set_xlabel("Long-Range Copy Recall  >= 64 tokens  (%, n = 4,137)", fontsize=10.8, labelpad=7)
    ax.set_ylabel("Validation CE  (nats, lower = better)", fontsize=10.8, labelpad=7)
    ax.tick_params(labelsize=9.8)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fig2_phase2_factorized(out_path: Path) -> None:
    """Phase-2 equal-scope factorized ablation: Transformer vs 3:1 Spectral-Reservoir vs 3:1 SSM+Res (0.10B)."""
    fig, ax = plt.subplots(figsize=(9.8, 4.15), dpi=300)

    labels = [
        "Clean\nBF16",
        "100%\n1.58b Only",
        "94% Equal\n1.58b+CIM6",
        "100%\nBF16+CIM6",
        "100% W4\nINT4+CIM6",
        "3:1 SSM+Res\nClean vs CIM6",
    ]
    x = np.arange(len(labels))
    w = 0.34

    # First 5 groups: Transformer (Ink) vs 3:1 Spectral-Reservoir (Teal)
    tf_vals = [3.769, 4.098, 4.093, 3.966, 4.087]
    res_vals = [3.712, 3.886, 3.977, 3.940, 3.969]

    for i in range(5):
        ax.bar(x[i] - w / 2, tf_vals[i], width=w * 0.88, color=INK, zorder=3)
        ax.bar(x[i] + w / 2, res_vals[i], width=w * 0.88, color=TEAL, zorder=3)
        ax.text(x[i] - w / 2, tf_vals[i] + 0.035, f"{tf_vals[i]:.2f}", ha="center", va="bottom", fontsize=8.8, color=INK)
        ax.text(
            x[i] + w / 2,
            res_vals[i] + 0.035,
            f"{res_vals[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.9,
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

    # 6th group: 3:1 SSM+Res Clean (3.783) vs 94% INT4+CIM6 (4.887, collapsed)
    ax.axvline(4.5, color="#D1D5DB", linestyle=":", linewidth=1.2)
    ax.bar(x[5] - w / 2, 3.783, width=w * 0.88, color=MUTED, zorder=3)
    ax.bar(x[5] + w / 2, 4.887, width=w * 0.88, color=CRIMSON, zorder=3)
    ax.text(x[5] - w / 2, 3.783 + 0.035, "3.78", ha="center", va="bottom", fontsize=8.8, color=MUTED)
    ax.text(x[5] + w / 2, 4.887 + 0.035, "4.89", ha="center", va="bottom", fontsize=9.1, fontweight="semibold", color=CRIMSON)

    ax.annotate(
        "Recurrent SSM collapses\n(+1.10 nats under CIM6)",
        xy=(x[5] + w / 2 - 0.06, 4.86),
        xytext=(3.78, 4.92),
        ha="center",
        fontsize=9.0,
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
    ax.set_xticklabels(labels, fontsize=9.3)
    ax.set_xlim(-0.58, 5.60)
    ax.set_ylim(3.35, 5.45)
    ax.set_ylabel("Validation CE  (nats, lower = better)", fontsize=10.8, labelpad=7)
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9.0)

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
    ax1.set_xticklabels(cats_j, fontsize=9.2)
    ax1.set_ylim(0, 10.6)
    ax1.set_ylabel("Valid Targets / sec  (K tok/s)", fontsize=10.2)
    ax1.set_title("JAX / XLA  (2x RTX 2080 Ti, FP32, Trace=1)", fontsize=10.2, pad=9, color=INK)
    ax1.legend(
        handles=[Patch(facecolor=INK, label="TF (A)"), Patch(facecolor=TEAL, label="Hybrid (B)"), Patch(facecolor=GREEN, label="HAT (C)")],
        loc="upper left",
        frameon=False,
        fontsize=8.9,
    )

    # Right: PyTorch 2x A6000 BF16
    cats_p = ["0.10B\nClean (B)", "0.10B\nHAT (C)", "0.26B ckpt\nClean (B)", "0.26B ckpt\nHAT (C)"]
    xp = np.arange(len(cats_p))
    wp = 0.35

    before = [47.14, 31.68, 11.72, 7.17]
    after = [51.20, 49.70, 25.07, 19.13]
    gains = ["1.09x", "1.57x", "2.14x", "2.67x"]

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
        ax2.text(xi + wp / 2, v * 0.50, g, ha="center", va="center", fontsize=8.3, color="white", fontweight="semibold", zorder=6)

    ax2.set_xticks(xp)
    ax2.set_xticklabels(cats_p, fontsize=9.0)
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Training Throughput  (K tok/s)", fontsize=10.2)
    ax2.set_title("PyTorch BF16  (2x RTX A6000, FlexAttn, L=2048)", fontsize=10.2, pad=9, color=INK)
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


def build_fig4_downstream_s2s_audio(out_path: Path) -> None:
    """Two-panel downstream offline S2S pilot: Part A (Speculative DFM + E5) & Part B (Real JP Audio)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.7, 3.95), dpi=300)

    # Left: Part A Step 1 Forward Calls (Commit-phase & Total, R2 2-seed) with E5 OOD miss callout
    cats_a = ["Commit-Phase\nForwards", "Total Episode\nForwards"]
    xa = np.arange(len(cats_a))
    wa = 0.25

    v_tf_ar = [6.19, 29.97]
    e_tf_ar = [0.43 / 2, 1.01 / 2]
    v_res_dfm = [3.18, 7.27]
    e_res_dfm = [0.47 / 2, 0.38 / 2]
    v_ssm_dfm = [3.13, 7.24]
    e_ssm_dfm = [0.32 / 2, 0.26 / 2]

    ax1.bar(xa - wa, v_tf_ar, width=wa * 0.88, yerr=e_tf_ar, capsize=3, color=INK, zorder=3)
    ax1.bar(xa, v_res_dfm, width=wa * 0.88, yerr=e_res_dfm, capsize=3, color=TEAL, zorder=3)
    ax1.bar(xa + wa, v_ssm_dfm, width=wa * 0.88, yerr=e_ssm_dfm, capsize=3, color=GREEN, zorder=3)

    for xi, vt, vr, vs in zip(xa, v_tf_ar, v_res_dfm, v_ssm_dfm):
        ax1.text(xi - wa, vt + 1.1, f"{vt:.2f}", ha="center", va="bottom", fontsize=8.6, color=INK, fontweight="semibold")
        ax1.text(xi, vr + 1.0, f"{vr:.2f}", ha="center", va="bottom", fontsize=8.6, color=TEAL, fontweight="semibold")
        ax1.text(xi + wa, vs + 1.0, f"{vs:.2f}", ha="center", va="bottom", fontsize=8.6, color=GREEN, fontweight="semibold")

    ax1.annotate(
        "4.1x fewer\n(25% KV)",
        xy=(1.0, 9.6),
        xytext=(1.13, 19.5),
        ha="center",
        fontsize=8.6,
        fontweight="semibold",
        color=TEAL,
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.05},
    )
    ax1.text(
        0.0,
        17.5,
        "E5 OOD Urgent Miss:\n91.7% -> 37.5% (Clean: 18.8%)",
        ha="center",
        va="center",
        fontsize=8.1,
        color=GREEN,
        fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F0FDFA", "edgecolor": "#CCFBF1", "lw": 0.9},
    )

    ax1.set_xticks(xa)
    ax1.set_xticklabels(cats_a, fontsize=9.4)
    ax1.set_xlim(-0.55, 1.48)
    ax1.set_ylim(0, 38.5)
    ax1.set_ylabel("Forward Passes / Episode  (lower = better)", fontsize=9.6)
    ax1.set_title("Part A: Speculative DFM + 5-Gate (E5, 2-seed)", fontsize=9.9, pad=8, color=INK)
    ax1.legend(
        handles=[
            Patch(facecolor=INK, label="TF AR (W4+CIM6)"),
            Patch(facecolor=TEAL, label="3:1 Res DFM (W4+CIM6)"),
            Patch(facecolor=GREEN, label="3:1 SSM+Res DFM (Clean)"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8.0,
    )

    # Right: Part B Real Japanese Audio (IngCrowd + open-noise, 0ms post-EOS, R2 2-seed AUROC)
    conds = ["Clean\nSpeech", "Field Noise\nSNR 10 dB", "Harsh + Crosstalk\nSNR 3 dB"]
    xb = np.arange(len(conds))
    wb = 0.19

    vad_e = [0.726, 0.667, 0.621]
    txt_o = [0.675, 0.675, 0.675]
    txt_e = [0.009, 0.009, 0.009]
    mimi_o = [0.726, 0.682, 0.669]
    mimi_e = [0.0055, 0.0045, 0.0255]
    fus_o = [0.711, 0.677, 0.658]
    fus_e = [0.0185, 0.013, 0.009]

    ax2.bar(xb - 1.5 * wb, vad_e, width=wb * 0.88, color="#D1D5DB", zorder=3)
    ax2.bar(xb - 0.5 * wb, txt_o, width=wb * 0.88, yerr=txt_e, capsize=2.5, color=AMBER, zorder=3)
    ax2.bar(xb + 0.5 * wb, mimi_o, width=wb * 0.88, yerr=mimi_e, capsize=2.5, color=TEAL, zorder=3)
    ax2.bar(xb + 1.5 * wb, fus_o, width=wb * 0.88, yerr=fus_e, capsize=2.5, color=GREEN, zorder=3)

    for xi in range(3):
        ax2.text(xb[xi] + 0.5 * wb, mimi_o[xi] + mimi_e[xi] + 0.008, f"{mimi_o[xi]:.2f}", ha="center", va="bottom", fontsize=8.2, color=TEAL, fontweight="semibold")
        ax2.text(xb[xi] - 0.5 * wb, txt_o[xi] + txt_e[xi] + 0.008, f"{txt_o[xi]:.2f}", ha="center", va="bottom", fontsize=8.0, color=AMBER)

    ax2.annotate(
        "+5.1 pt vs 0%-WER Text\n(0 ms wait, zero ASR)",
        xy=(0.5 * wb, 0.755),
        xytext=(0.45, 0.822),
        ha="center",
        fontsize=8.3,
        fontweight="semibold",
        color=TEAL,
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.0},
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels(conds, fontsize=9.1)
    ax2.set_ylim(0.50, 0.88)
    ax2.set_ylabel("Floor Hold vs. Shift AUROC  (higher = better)", fontsize=9.8)
    ax2.set_title("Part B: Real JP Audio (IngCrowd, n=196, 2-seed)", fontsize=10.0, pad=9, color=INK)
    ax2.legend(
        handles=[
            Patch(facecolor="#D1D5DB", label="Energy VAD (+800ms)"),
            Patch(facecolor=AMBER, label="0%-WER Text Only"),
            Patch(facecolor=TEAL, label="Mimi 12.5 Hz (0 ASR)"),
            Patch(facecolor=GREEN, label="Mimi+Text (W4+CIM6)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=7.9,
    )

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "docs" / "figs"
    for old in ("fig1_pilot_tradeoff.pdf", "fig2_systems_throughput.pdf", "fig2_cim_resilience.pdf"):
        p = fig_dir / old
        if p.exists():
            p.unlink()
    build_fig1_copy_vs_ce(fig_dir / "fig1_copy_vs_ce.pdf")
    build_fig2_phase2_factorized(fig_dir / "fig2_phase2_factorized.pdf")
    build_fig3_systems_speed(fig_dir / "fig3_systems_speed.pdf")
    build_fig4_downstream_s2s_audio(fig_dir / "fig4_downstream_s2s_audio.pdf")
    print(f"Generated v0.4.0 minimalist Keynote figures under {fig_dir}")


if __name__ == "__main__":
    main()
