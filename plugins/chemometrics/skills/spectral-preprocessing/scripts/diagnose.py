#!/usr/bin/env python
"""Diagnose a preprocessing pipeline step by step.

Runs the spec one step at a time and reports, after each, the output shape and
basic statistics plus any red flags (non-finite values, all-zero rows,
near-constant rows, sign flips). This catches the common failures — a range cut
that drops all features, a scatter correction that produces NaNs, an order
mistake that zeroes the signal — without needing to plot.

Usage::

    python diagnose.py --input spectra.csv --spec spec.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pipeline import build_pipeline, load_spec  # noqa: E402
from load_spectra import load_spectra  # noqa: E402


def _flags(X: np.ndarray) -> list[str]:
    flags = []
    n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
    if n_bad:
        flags.append(f"{n_bad} non-finite")
    row_std = np.nanstd(X, axis=1)
    if np.any(row_std == 0):
        flags.append(f"{int(np.sum(row_std == 0))} constant row(s)")
    if np.all(X == 0):
        flags.append("ALL ZERO")
    return flags


def _describe(name: str, X: np.ndarray) -> str:
    flags = _flags(X)
    tag = ("  [" + ", ".join(flags) + "]") if flags else ""
    return (
        f"{name:<28} shape={str(X.shape):<16} "
        f"mean={np.nanmean(X):+.3g} std={np.nanstd(X):.3g}{tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    X, x_axis = load_spectra(args.input)
    print(_describe("input", X))

    spec = load_spec(args.spec)
    # Build the full pipeline once (validates the spec), then apply cumulatively.
    Xi = X
    for i, entry in enumerate(spec):
        sub = build_pipeline(spec[: i + 1], x_axis=x_axis)
        Xi = sub.fit_transform(X)
        print(_describe(entry["type"], Xi))

    # Optional richer diagnosis via the (experimental) PreprocessingInspector.
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from chemotools.inspector import PreprocessingInspector  # noqa: F401
        print("\n(chemotools.inspector.PreprocessingInspector is available for "
              "interactive per-step visualization.)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
