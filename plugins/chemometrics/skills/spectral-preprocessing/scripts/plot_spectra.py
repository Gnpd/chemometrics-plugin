#!/usr/bin/env python
"""Overlay raw vs. preprocessed spectra to eyeball what a pipeline did.

Requires matplotlib (``pip install chemotools[viz]`` or ``pip install matplotlib``).
If matplotlib is missing it prints an actionable message and a text summary
instead of crashing.

Usage::

    python plot_spectra.py --input spectra.csv --spec spec.yaml --out compare.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pipeline import build_pipeline, load_spec  # noqa: E402
from load_spectra import load_spectra  # noqa: E402


def _text_summary(raw: np.ndarray, proc: np.ndarray) -> str:
    return (
        "matplotlib not installed — install with: pip install chemotools[viz]\n"
        f"raw  : shape {raw.shape}, mean {np.nanmean(raw):.4g}, std {np.nanstd(raw):.4g}\n"
        f"proc : shape {proc.shape}, mean {np.nanmean(proc):.4g}, std {np.nanstd(proc):.4g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", help="Save figure here (e.g. compare.png). Else show().")
    parser.add_argument("--max-spectra", type=int, default=30, help="Rows to plot.")
    args = parser.parse_args()

    X, x_axis = load_spectra(args.input)
    pipeline = build_pipeline(load_spec(args.spec), x_axis=x_axis)
    X_proc = pipeline.fit_transform(X)

    try:
        import matplotlib

        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(_text_summary(X, X_proc))
        return

    k = min(args.max_spectra, X.shape[0])
    ax_raw = x_axis if (x_axis is not None and x_axis.shape[0] == X.shape[1]) else np.arange(X.shape[1])
    ax_proc = x_axis if (x_axis is not None and x_axis.shape[0] == X_proc.shape[1]) else np.arange(X_proc.shape[1])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(ax_raw, X[:k].T, lw=0.6)
    ax0.set_title(f"Raw ({k} spectra)")
    ax1.plot(ax_proc, X_proc[:k].T, lw=0.6)
    ax1.set_title("Preprocessed")
    for ax in (ax0, ax1):
        ax.set_xlabel("x-axis (wavenumber/wavelength or index)")
        ax.set_ylabel("intensity")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
