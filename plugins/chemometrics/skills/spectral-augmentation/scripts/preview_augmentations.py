#!/usr/bin/env python
"""Preview each augmentation type on sample spectra to check plausibility.

Applies each chemotools augmentation transformer (with a chosen magnitude) to a
few spectra and reports how much it perturbed them (mean absolute change,
correlation to the original), plus an optional overlay figure. Use this to pick
augmentation magnitudes that stay within realistic variation — augmenting beyond
what the instrument/sample could actually produce teaches the model noise.

Usage::

    python preview_augmentations.py --input spectra.csv
    python preview_augmentations.py --input spectra.npy --scale 0.02 --shift 3 --out aug.png
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


def _options(scale: float, shift: int, sigma: float, seed: int) -> dict:
    from chemotools.augmentation import (
        AddNoise,
        BaselineShift,
        FractionalShift,
        GaussianBroadening,
        IndexShift,
        SpectrumScale,
    )

    return {
        "add_noise": AddNoise(distribution="gaussian", scale=scale, random_state=seed),
        "baseline_shift": BaselineShift(scale=scale, random_state=seed),
        "spectrum_scale": SpectrumScale(scale=scale, random_state=seed),
        "gaussian_broadening": GaussianBroadening(sigma=sigma, random_state=seed),
        "index_shift": IndexShift(shift=shift, random_state=seed),
        "fractional_shift": FractionalShift(shift=float(shift), random_state=seed),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.ravel(), b.ravel()
    if np.std(af) == 0 or np.std(bf) == 0:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--scale", type=float, default=0.02, help="Noise/shift/scale magnitude.")
    parser.add_argument("--shift", type=int, default=3, help="Index/fractional shift.")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian broadening sigma.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Save overlay figure (needs matplotlib).")
    parser.add_argument("--max-spectra", type=int, default=8)
    args = parser.parse_args()

    X = _load_X(args.input)
    opts = _options(args.scale, args.shift, args.sigma, args.seed)

    print(f"X={X.shape}  scale={args.scale}  shift={args.shift}  sigma={args.sigma}\n")
    print(f"{'augmentation':<22}  {'mean_abs_change':>15}  {'corr_to_original':>17}")
    results = {}
    for name, t in opts.items():
        Xa = t.fit_transform(X)
        results[name] = Xa
        mac = float(np.mean(np.abs(Xa - X)))
        print(f"{name:<22}  {mac:>15.4g}  {_corr(X, Xa):>17.4f}")

    print(
        "\nAim for high corr_to_original with a visible-but-realistic change. "
        "corr near 1.0 = too weak; low corr = likely beyond physical plausibility."
    )

    if args.out:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed; skipped figure.)")
            return
        k = min(args.max_spectra, X.shape[0])
        n = len(results)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        for ax, (name, Xa) in zip(axes.ravel(), results.items()):
            ax.plot(X[:k].T, lw=0.5, color="0.7")
            ax.plot(Xa[:k].T, lw=0.6)
            ax.set_title(name, fontsize=10)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}  (grey = original, colored = augmented)")


if __name__ == "__main__":
    main()
