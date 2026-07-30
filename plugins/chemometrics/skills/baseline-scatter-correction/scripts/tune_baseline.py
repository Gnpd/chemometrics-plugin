#!/usr/bin/env python
"""Sweep the regularization of a reweighted baseline and report the tradeoff.

For AirPLS / ArPLS / AsLs the key knob is ``lam`` (smoothness). Too small and
the baseline chases the peaks (over-subtraction, negative signal); too large and
it is too stiff to follow real drift (residual baseline). This sweeps ``lam``
over a grid and reports, per value:

* ``baseline_roughness`` — mean squared 2nd difference of the estimated baseline
  (lower = smoother/stiffer);
* ``neg_fraction`` — fraction of corrected points below zero (higher = more
  over-subtraction into the peaks);
* ``residual_offset`` — mean of the corrected signal (should sit near zero once
  the baseline is removed).

Pick the smallest ``lam`` that keeps ``neg_fraction`` low while
``residual_offset`` stays near zero — that is the least-stiff baseline that
doesn't eat your peaks.

Usage::

    python tune_baseline.py --input spectra.csv --method airpls
    python tune_baseline.py --input spectra.npy --method arpls --lams 1e3,1e4,1e5,1e6
"""

from __future__ import annotations

import argparse

import numpy as np

METHODS = {
    "airpls": ("chemotools.baseline", "AirPls"),
    "arpls": ("chemotools.baseline", "ArPls"),
    "asls": ("chemotools.baseline", "AsLs"),
}


def _load_X(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        X = np.load(path)
    else:
        import pandas as pd

        X = pd.read_csv(path).to_numpy()
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(1, -1) if X.ndim == 1 else X


def _make(method: str, lam: float):
    import importlib

    module_name, class_name = METHODS[method]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(lam=lam)


def evaluate(X: np.ndarray, method: str, lam: float) -> dict[str, float]:
    t = _make(method, lam)
    corrected = t.fit_transform(X)
    baseline = X - corrected
    roughness = float(np.mean(np.diff(baseline, n=2, axis=1) ** 2))
    neg_fraction = float(np.mean(corrected < 0))
    residual_offset = float(np.mean(corrected))
    return {
        "lam": lam,
        "baseline_roughness": roughness,
        "neg_fraction": neg_fraction,
        "residual_offset": residual_offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help=".csv or .npy spectra")
    parser.add_argument("--method", choices=sorted(METHODS), default="airpls")
    parser.add_argument(
        "--lams",
        default="1e2,1e3,1e4,1e5,1e6,1e7",
        help="Comma-separated lam grid.",
    )
    args = parser.parse_args()

    X = _load_X(args.input)
    lams = [float(v) for v in args.lams.split(",")]

    print(f"method={args.method}  X={X.shape}\n")
    print(f"{'lam':>10}  {'roughness':>12}  {'neg_frac':>9}  {'offset':>10}")
    for lam in lams:
        r = evaluate(X, args.method, lam)
        print(
            f"{r['lam']:>10.0e}  {r['baseline_roughness']:>12.4g}  "
            f"{r['neg_fraction']:>9.3f}  {r['residual_offset']:>10.3g}"
        )
    print(
        "\nHeuristic: smallest lam with low neg_frac and near-zero offset. "
        "Verify by plotting the corrected spectra."
    )


if __name__ == "__main__":
    main()
