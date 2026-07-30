#!/usr/bin/env python
"""Resample spectra onto a common x-axis grid using XAxisInterpolator.

Spectra measured on different wavenumber/wavelength grids (multiple instruments,
drifted calibration) cannot be stacked into one matrix until they share a grid.
This resamples every spectrum onto a common grid via chemotools'
``XAxisInterpolator``, using scikit-learn metadata routing to pass each
spectrum's own axis.

The x-axis is taken from (in priority order): ``--x-axis`` file, else the CSV
float column headers (shared grid). Provide a 2-D ``--x-axis`` (one row per
spectrum) for per-sample grids.

Usage::

    # shared grid from CSV header, resample to 600 points over the data range
    python align_axes.py --input spectra.csv --num 600 --out aligned.csv

    # per-sample grids from a separate file, explicit common range
    python align_axes.py --input spectra.npy --x-axis axes.npy \
        --start 650 --end 1550 --num 1000 --method linear --fill 0 --out aligned.npy
"""

from __future__ import annotations

import argparse

import numpy as np


def _load(path: str, header: bool = True):
    if path.endswith(".npy"):
        return np.load(path), None
    import pandas as pd

    df = pd.read_csv(path, header=0 if header else None)
    X = df.to_numpy(dtype=float)
    try:
        axis = np.asarray([float(c) for c in df.columns], dtype=float)
    except (TypeError, ValueError):
        axis = None
    return X, axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--x-axis", help="Per-sample (2-D) or shared (1-D) axis file (.npy/.csv).")
    parser.add_argument("--start", type=float, help="Common grid start (default: data min).")
    parser.add_argument("--end", type=float, help="Common grid end (default: data max).")
    parser.add_argument("--num", type=int, default=1000, help="Common grid points.")
    parser.add_argument("--method", choices=["linear", "cubic", "pchip"], default="linear")
    parser.add_argument("--fill", type=float, default=0.0, help="Fill value outside input grid.")
    parser.add_argument("--out", help="Write aligned spectra (.csv/.npy).")
    args = parser.parse_args()

    X, axis_from_header = _load(args.input)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if args.x_axis:
        x_axis, _ = _load(args.x_axis, header=False)
        x_axis = np.asarray(x_axis, dtype=float)
    elif axis_from_header is not None:
        x_axis = axis_from_header
    else:
        raise SystemExit("No x-axis: pass --x-axis or use a CSV with float column headers.")

    lo = args.start if args.start is not None else float(np.nanmin(x_axis))
    hi = args.end if args.end is not None else float(np.nanmax(x_axis))
    x_common = np.linspace(lo, hi, args.num)

    import sklearn
    from chemotools.adaptation import XAxisInterpolator

    sklearn.set_config(enable_metadata_routing=True)
    interp = (
        XAxisInterpolator(common_x_axis=x_common, method=args.method, left=args.fill, right=args.fill)
        .set_fit_request(x_axis=True)
        .set_transform_request(x_axis=True)
    )
    X_aligned = interp.fit_transform(X, x_axis=x_axis)

    n_bad = int(np.isnan(X_aligned).sum() + np.isinf(X_aligned).sum())
    print(f"{X.shape} on axis [{lo:.4g}, {hi:.4g}] -> {X_aligned.shape} common grid")
    print(f"method={args.method}  fill={args.fill}  non-finite={n_bad}")
    if n_bad:
        print("WARNING: non-finite values — common grid extends past some input grids; "
              "narrow --start/--end or change --fill.")

    if args.out:
        if args.out.endswith(".npy"):
            np.save(args.out, X_aligned)
        else:
            import pandas as pd

            pd.DataFrame(X_aligned, columns=x_common).to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
