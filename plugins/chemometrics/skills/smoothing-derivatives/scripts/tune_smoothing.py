#!/usr/bin/env python
"""Sweep Savitzky-Golay smoothing settings and report noise vs. peak retention.

Smoothing trades noise reduction against resolution: a wider window (or lower
polynomial order) removes more noise but also flattens and broadens peaks. This
sweeps ``window_length`` for a given ``polyorder`` and reports:

* ``noise_proxy`` — median absolute lag-1 difference (a high-frequency noise
  estimate); lower after smoothing is the point;
* ``noise_reduction`` — raw noise_proxy / smoothed noise_proxy (higher = more
  denoising);
* ``peak_retention`` — mean per-spectrum peak-to-trough range after / before
  (near 1.0 = peaks preserved; well below 1.0 = over-smoothed).

Pick the smallest window that gives adequate ``noise_reduction`` while keeping
``peak_retention`` high (rule of thumb: window narrower than your narrowest peak).

Usage::

    python tune_smoothing.py --input spectra.csv
    python tune_smoothing.py --input spectra.npy --polyorder 2 --windows 5,9,15,21,31
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


def _peak_range(X: np.ndarray) -> float:
    return float(np.mean(np.nanmax(X, axis=1) - np.nanmin(X, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--windows", default="5,9,15,21,31,41")
    args = parser.parse_args()

    from chemotools.smooth import SavitzkyGolayFilter

    X = _load_X(args.input)
    raw_noise = _noise_proxy(X)
    raw_range = _peak_range(X)
    windows = [int(w) for w in args.windows.split(",")]

    print(f"X={X.shape}  polyorder={args.polyorder}  raw_noise={raw_noise:.4g}\n")
    print(f"{'window':>7}  {'noise_proxy':>12}  {'noise_reduction':>15}  {'peak_retention':>15}")
    for w in windows:
        if w % 2 == 0 or w <= args.polyorder:
            print(f"{w:>7}  {'(skip: need odd window > polyorder)':>46}")
            continue
        Xs = SavitzkyGolayFilter(window_length=w, polyorder=args.polyorder).fit_transform(X)
        ns = _noise_proxy(Xs)
        reduction = raw_noise / ns if ns > 0 else float("inf")
        retention = _peak_range(Xs) / raw_range if raw_range > 0 else float("nan")
        print(f"{w:>7}  {ns:>12.4g}  {reduction:>15.2f}  {retention:>15.3f}")

    print(
        "\nHeuristic: smallest window with enough noise_reduction and "
        "peak_retention ~1.0. Confirm visually."
    )


if __name__ == "__main__":
    main()
