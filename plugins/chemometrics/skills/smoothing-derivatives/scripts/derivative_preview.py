#!/usr/bin/env python
"""Preview 0th/1st/2nd Savitzky-Golay derivatives of spectra.

Derivatives remove baselines and sharpen overlapping bands but amplify noise —
the effect is best judged by eye. This applies a Savitzky-Golay derivative at
orders 0 (smoothing only), 1, and 2 with the chosen window/polyorder, reports a
noise proxy for each, and optionally saves a 3-panel figure.

Guidance:
* 1st derivative removes a constant additive baseline and sharpens peaks.
* 2nd derivative also removes a linear baseline and resolves overlapping bands,
  but roughly squares the noise amplification — needs a larger smoothing window.
* ``polyorder`` must exceed ``deriv`` (use polyorder >= 2 for a 2nd derivative).

Usage::

    python derivative_preview.py --input spectra.csv --window 21 --polyorder 2
    python derivative_preview.py --input spectra.npy --out deriv.png
"""

from __future__ import annotations

import argparse

import numpy as np


def _load_X(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        X = np.load(path)
    else:
        import pandas as pd

        X = pd.read_csv(path).to_numpy()
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(1, -1) if X.ndim == 1 else X


def _noise_proxy(X: np.ndarray) -> float:
    d = np.diff(X, axis=1)
    return float(np.median(np.abs(d - np.median(d))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--out", help="Save 3-panel figure (needs matplotlib).")
    parser.add_argument("--max-spectra", type=int, default=20)
    args = parser.parse_args()

    from chemotools.derivative import SavitzkyGolay

    X = _load_X(args.input)
    results = {}
    for deriv in (0, 1, 2):
        t = SavitzkyGolay(window_length=args.window, polyorder=args.polyorder, deriv=deriv)
        results[deriv] = t.fit_transform(X)

    print(f"X={X.shape}  window={args.window}  polyorder={args.polyorder}\n")
    print(f"{'deriv':>6}  {'noise_proxy':>12}  {'std':>12}")
    for deriv, Xd in results.items():
        print(f"{deriv:>6}  {_noise_proxy(Xd):>12.4g}  {np.nanstd(Xd):>12.4g}")

    if args.out:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed; skipped figure.)")
            return
        k = min(args.max_spectra, X.shape[0])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, deriv in zip(axes, (0, 1, 2)):
            ax.plot(results[deriv][:k].T, lw=0.6)
            ax.set_title(f"deriv={deriv}" + (" (smoothed)" if deriv == 0 else ""))
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
