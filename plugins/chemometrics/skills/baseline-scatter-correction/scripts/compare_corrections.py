#!/usr/bin/env python
"""Apply several baseline/scatter corrections to the same spectra and compare.

Runs a menu of correction options and reports, per option, how much per-spectrum
offset/scale variation it removed — the thing baseline and scatter corrections
exist to reduce. Optionally saves a multi-panel figure.

Metrics:
* ``between_spectra_std`` — mean over channels of the across-spectra standard
  deviation. Scatter/baseline effects inflate this; a good correction lowers it
  (for scatter effects) without flattening real chemical differences.
* ``mean_row_range`` — mean per-spectrum peak-to-trough range (sanity: peaks not
  destroyed).

Usage::

    python compare_corrections.py --input spectra.csv
    python compare_corrections.py --input spectra.npy --out compare.png
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


def _options() -> dict[str, object]:
    from chemotools.baseline import AirPls, ArPls, LinearCorrection
    from chemotools.scatter import (
        MultiplicativeScatterCorrection,
        RobustNormalVariate,
        StandardNormalVariate,
    )

    return {
        "raw": None,
        "linear_correction": LinearCorrection(),
        "airpls(lam=1e5)": AirPls(lam=1e5),
        "arpls(lam=1e5)": ArPls(lam=1e5),
        "snv": StandardNormalVariate(),
        "rnv": RobustNormalVariate(),
        "msc": MultiplicativeScatterCorrection(),
    }


def _metrics(X: np.ndarray) -> tuple[float, float]:
    between = float(np.mean(np.nanstd(X, axis=0)))
    row_range = float(np.mean(np.nanmax(X, axis=1) - np.nanmin(X, axis=1)))
    return between, row_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", help="Save a comparison figure (needs matplotlib).")
    parser.add_argument("--max-spectra", type=int, default=30)
    args = parser.parse_args()

    X = _load_X(args.input)
    options = _options()
    results = {}
    for name, t in options.items():
        Xt = X if t is None else t.fit_transform(X)
        results[name] = Xt

    print(f"X={X.shape}\n")
    print(f"{'option':<22}  {'between_spectra_std':>19}  {'mean_row_range':>15}")
    for name, Xt in results.items():
        b, r = _metrics(Xt)
        print(f"{name:<22}  {b:>19.4g}  {r:>15.4g}")

    if args.out:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed; skipped figure. pip install matplotlib)")
            return
        k = min(args.max_spectra, X.shape[0])
        n = len(results)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        for ax, (name, Xt) in zip(axes.ravel(), results.items()):
            ax.plot(Xt[:k].T, lw=0.5)
            ax.set_title(name, fontsize=10)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
