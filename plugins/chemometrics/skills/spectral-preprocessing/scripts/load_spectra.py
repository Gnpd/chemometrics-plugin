#!/usr/bin/env python
"""Load spectra into the ``(n_samples, n_features)`` float array chemotools expects.

Handles the two shape mistakes that break chemotools/scikit-learn transformers:

* a single spectrum arriving as a 1-D array (reshaped to ``(1, n_features)``);
* the x-axis (wavenumbers/wavelengths) being lost because it lived in the
  CSV header row — this reads it back out as a float array.

Supported inputs: ``.csv`` (first row = x-axis header by default), ``.parquet``,
and ``.npy``. Returns ``(X, x_axis)`` where ``x_axis`` is ``None`` when it cannot
be recovered.

CLI usage (prints a shape/quality report)::

    python load_spectra.py --input spectra.csv
"""

from __future__ import annotations

import argparse

import numpy as np


def load_spectra(path: str, *, header: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Read spectra from *path*.

    Parameters
    ----------
    path : str
        ``.csv``, ``.parquet``, or ``.npy`` file.
    header : bool, default=True
        For CSV/parquet: treat column names as the x-axis. Column labels that
        parse as floats (the chemotools dataset convention) become ``x_axis``.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), dtype float64
    x_axis : np.ndarray of shape (n_features,) or None
    """
    x_axis: np.ndarray | None = None

    if path.endswith(".npy"):
        X = np.load(path)
    elif path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        X = df.to_numpy()
        x_axis = _axis_from_columns(df.columns) if header else None
    elif path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path, header=0 if header else None)
        X = df.to_numpy()
        x_axis = _axis_from_columns(df.columns) if header else None
    else:
        raise ValueError(f"Unsupported extension for {path!r}. Use .csv/.parquet/.npy.")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix after loading; got shape {X.shape}.")

    if x_axis is not None and x_axis.shape[0] != X.shape[1]:
        x_axis = None  # header wasn't really an axis
    return X, x_axis


def _axis_from_columns(columns) -> np.ndarray | None:
    try:
        return np.asarray([float(c) for c in columns], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def report(X: np.ndarray, x_axis: np.ndarray | None) -> str:
    """Human-readable quality report used by the CLI and diagnostics."""
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    lines = [
        f"shape         : {X.shape}  (n_samples, n_features)",
        f"dtype         : {X.dtype}",
        f"intensity min : {np.nanmin(X):.6g}",
        f"intensity max : {np.nanmax(X):.6g}",
        f"NaN / Inf     : {n_nan} / {n_inf}",
    ]
    if x_axis is not None:
        ascending = bool(np.all(np.diff(x_axis) > 0))
        lines.append(
            f"x_axis        : {x_axis.shape} from {x_axis[0]:.6g} to {x_axis[-1]:.6g} "
            f"({'ascending' if ascending else 'descending/irregular'})"
        )
    else:
        lines.append("x_axis        : not recovered (pass indices explicitly downstream)")
    if n_nan or n_inf:
        lines.append("WARNING: non-finite values present — clean before fitting a model.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not treat the first row / column names as an x-axis.",
    )
    args = parser.parse_args()
    X, x_axis = load_spectra(args.input, header=not args.no_header)
    print(report(X, x_axis))


if __name__ == "__main__":
    main()
