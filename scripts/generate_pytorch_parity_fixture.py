"""Generate deterministic FP32 PyTorch reference outputs for cross-framework operator parity tests.

Run once in any environment with `torch` installed:
    python scripts/generate_pytorch_parity_fixture.py

This saves `tests/fixtures/pytorch_operator_ref_fp32.npz` (~30 KB) so `pytest` verifies
`<1e-6` JAX-vs-PyTorch operator parity even in JAX-only CPU/TPU CI environments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def main() -> None:
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)

    # 1. Hankel spectral filters (FP64 eigh -> FP32)
    L, K, C = 64, 8, 16
    idx = torch.arange(1, L + 1, dtype=torch.float64)
    s = idx[:, None] + idx[None, :]
    z = 2.0 / (s**3 - s)
    eigvals, eigvecs = torch.linalg.eigh(z)
    top_vals = torch.clamp(torch.flip(eigvals[-K:], dims=[0]), min=1e-12)
    top_vecs = torch.flip(eigvecs[:, -K:], dims=[1]).T  # [K, L]
    max_idx = torch.argmax(top_vecs.abs(), dim=1)
    signs = torch.where(top_vecs[torch.arange(K), max_idx] < 0.0, -1.0, 1.0).to(torch.float64)
    top_vecs = top_vecs * signs[:, None]
    phi_pos = (top_vecs * (top_vals**0.25)[:, None]).to(torch.float32)
    alt = torch.where((torch.arange(L) % 2) == 0, 1.0, -1.0).to(torch.float32)
    phi_neg = phi_pos * alt[None, :]

    # 2. Contracted causal 1D FFT convolution
    x_np = rng.standard_normal((2, L, C)).astype(np.float32)
    wp_np = (rng.standard_normal((K, C)) * 0.2).astype(np.float32)
    wn_np = (rng.standard_normal((K, C)) * 0.2).astype(np.float32)
    xt = torch.from_numpy(x_np)
    wpt = torch.from_numpy(wp_np)
    wnt = torch.from_numpy(wn_np)
    n_fft = 1 << (2 * L - 1).bit_length()
    x_fft = torch.fft.rfft(xt, n=n_fft, dim=1)
    fp = torch.fft.rfft(phi_pos, n=n_fft, dim=1)
    fn = torch.fft.rfft(phi_neg, n=n_fft, dim=1)
    h_real = fp.real.T @ wpt + fn.real.T @ wnt
    h_imag = fp.imag.T @ wpt + fn.imag.T @ wnt
    h_fft = torch.complex(h_real, h_imag)
    y_fft = torch.fft.irfft(x_fft * h_fft[None, :, :], n=n_fft, dim=1)[:, :L, :]

    # 3. 1.58-bit absmean ternary STE & W4 symmetric per-row INT4 STE
    w_np = rng.standard_normal((16, 32)).astype(np.float32)
    wt = torch.from_numpy(w_np)
    scale = wt.abs().mean().clamp(min=1e-6)
    wq = torch.clamp(torch.round(wt / scale), -1.0, 1.0) * scale
    scale_i4 = (wt.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-6)
    wq_int4 = torch.clamp(torch.round(wt / scale_i4), -7.0, 7.0) * scale_i4

    # 4. 6-bit RMS-calibrated ADC STE
    qmax = float((1 << 5) - 1)
    rms = torch.sqrt(xt.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    step = (rms * 4.0).clamp(min=1e-6) / qmax
    xq = torch.clamp(torch.round(xt / step), -qmax, qmax) * step

    # 5. 5-step quintic Newton-Schulz (FP32)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = wt.clone()
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    out_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pytorch_operator_ref_fp32.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        torch_version=np.asarray(torch.__version__),
        x_in=x_np,
        w_in=w_np,
        w_pos=wp_np,
        w_neg=wn_np,
        phi_pos_ref=phi_pos.numpy(),
        phi_neg_ref=phi_neg.numpy(),
        y_fft_ref=y_fft.numpy(),
        w_ternary_ref=wq.numpy(),
        w_int4_ref=wq_int4.numpy(),
        x_adc6_ref=xq.numpy(),
        w_ns5_ref=X.numpy(),
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes, torch={torch.__version__})")


if __name__ == "__main__":
    main()
