#!/usr/bin/env python
"""Demonstrate instrument standardization (DS / PDS) end to end.

Self-contained: synthesizes spectra for a "source" and a "target" instrument
that differ by a smooth per-channel gain plus offset (a stand-in for real
instrument response differences), then uses a small set of *transfer standards*
(samples measured on both instruments) to learn a map that brings target-instrument
spectra onto the source instrument's response. Reports the mean per-spectrum RMSE
between instruments before and after transfer on a held-out validation set.

Direct Standardization (DS) learns one global linear map; Piecewise Direct
Standardization (PDS) learns local windowed maps and usually transfers better
when the instrument difference varies across the spectrum.

Run with your own data by adapting the ``fit(X_target, X_source=X_source)``
calls — you need transfer standards measured on both instruments.

Usage::

    python transfer_demo.py
    python transfer_demo.py --n-standards 15 --window 25
"""

from __future__ import annotations

import argparse

import numpy as np


def _make_instrument_pair(rng, n_samples, n_features):
    """Return (source, target) spectra: same chemistry, different instrument."""
    x = np.linspace(0, 1, n_features)
    source = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # 3 Gaussian bands with sample-varying heights (the "chemistry")
        for center in (0.25, 0.5, 0.75):
            height = rng.uniform(0.5, 1.5)
            width = 0.03
            source[i] += height * np.exp(-0.5 * ((x - center) / width) ** 2)
    # instrument difference: smooth multiplicative gain + additive offset
    gain = 1.0 + 0.4 * np.sin(2 * np.pi * x)
    offset = 0.05 * x
    target = source * gain + offset
    return source, target


def _rmse_between(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((A - B) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-standards", type=int, default=20, help="Transfer standards.")
    parser.add_argument("--n-val", type=int, default=50, help="Validation spectra.")
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--window", type=int, default=25, help="PDS window length.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src_std, tgt_std = _make_instrument_pair(rng, args.n_standards, args.n_features)
    src_val, tgt_val = _make_instrument_pair(rng, args.n_val, args.n_features)

    from chemotools.adaptation import (
        DirectStandardization,
        PiecewiseDirectStandardization,
    )

    before = _rmse_between(tgt_val, src_val)
    print(f"features={args.n_features}  transfer standards={args.n_standards}\n")
    print(f"{'method':<10}  {'val RMSE (target vs source)':>28}")
    print(f"{'none':<10}  {before:>28.5g}")

    # DS: learn map on transfer standards, apply to validation target spectra.
    ds = DirectStandardization().fit(tgt_std, X_source=src_std)
    ds_val = ds.transform(tgt_val)
    print(f"{'DS':<10}  {_rmse_between(ds_val, src_val):>28.5g}")

    pds = PiecewiseDirectStandardization(window_length=args.window).fit(
        tgt_std, X_source=src_std
    )
    pds_val = pds.transform(tgt_val)
    print(f"{'PDS':<10}  {_rmse_between(pds_val, src_val):>28.5g}")

    print(
        "\nLower is better: transfer should shrink the target-vs-source gap. "
        "PDS often beats DS when the instrument difference varies across the axis."
    )


if __name__ == "__main__":
    main()
