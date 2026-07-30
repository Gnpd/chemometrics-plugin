#!/usr/bin/env python
"""Supervised band selection from a fitted PLS model — VIP or SR.

Both scores rank wavelengths/wavenumbers by their contribution to a fitted PLS:

* **VIP** (Variable Importance in Projection) — a per-band score whose mean
  square is 1, so the ``>1`` convention gives a natural threshold.
* **SR** (Selectivity Ratio) — explained/residual variance per band from the
  PLS target-projection; peaks are sharper than VIP but there is no universal
  cut-off (use a percentile or an F-test threshold).

These are chemometrics-specific (chemotools ``feature_selection``); there is no
sklearn equivalent. They need a **fitted PLS**, so fit the model first
(``fit_model.py``) at the LV count chosen by CV (``select_components.py``).
For a PLS-DA model the underlying multi-output PLS is used.

Importable (:func:`importance`, :func:`top_bands`) and CLI-runnable::

    python feature_importance.py --dataset fermentation --n-components 6 --method vip --top 15
    python feature_importance.py --dataset coffee --task classification --method sr --plot bands.png
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _fit_model_module():
    path = Path(__file__).resolve().parent / "fit_model.py"
    spec = importlib.util.spec_from_file_location("_cm_fit_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _underlying_pls(model):
    """Return the fitted chemotools/sklearn PLS from a model or PLSDA wrapper."""
    return getattr(model, "pls_", model)


def importance(model, X, y, *, method: str = "vip", threshold: float = 1.0) -> np.ndarray:
    """Per-band importance scores for a fitted PLS (or PLSDA) *model*.

    Parameters
    ----------
    model : fitted PLSRegression or PLSDA
    method : ``"vip"`` or ``"sr"``.
    threshold : passed to the selector (only affects its ``support_mask_``;
        the returned array is the raw per-band score).
    """
    from chemotools.feature_selection import SRSelector, VIPSelector

    pls = _underlying_pls(model)
    Y = _matches_pls_target(model, y)
    selector_cls = {"vip": VIPSelector, "sr": SRSelector}[method]
    selector = selector_cls(pls, threshold=threshold).fit(X, Y)
    return np.asarray(selector.feature_scores_, dtype=np.float64)


def _matches_pls_target(model, y):
    """PLS-DA fits on a one-hot target; VIP/SR must see the same shape."""
    if hasattr(model, "_lb"):  # a fitted PLSDA
        Y = model._lb.transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])
        return Y
    return y


def top_bands(scores: np.ndarray, x_axis=None, *, top: int = 10):
    """Indices (and x-axis positions) of the ``top`` highest-scoring bands."""
    order = np.argsort(scores)[::-1][:top]
    positions = None if x_axis is None else np.asarray(x_axis)[order]
    return order, positions


def _plot(scores, x_axis, method, threshold, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot; pip install matplotlib)")
        return
    xs = np.arange(len(scores)) if x_axis is None else np.asarray(x_axis)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, scores, lw=0.8)
    if method == "vip":
        ax.axhline(threshold, color="tab:red", ls="--", label=f"threshold={threshold}")
        ax.legend()
    ax.set_xlabel("wavenumber / wavelength" if x_axis is not None else "feature index")
    ax.set_ylabel(f"{method.upper()} score")
    ax.set_title(f"{method.upper()} band importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["fermentation", "coffee"])
    src.add_argument("--input")
    parser.add_argument("--y")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--method", choices=["vip", "sr"], default="vip")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--plot")
    args = parser.parse_args()

    fm = _fit_model_module()
    if args.dataset:
        X, y, x_axis = fm.load_dataset(args.dataset)
    else:
        X, y, x_axis = fm._load_xy(args)

    model = fm.build_model(args.task, n_components=args.n_components)
    model.fit(X, y)
    scores = importance(model, X, y, method=args.method, threshold=args.threshold)
    order, positions = top_bands(scores, x_axis, top=args.top)

    print(f"{args.method.upper()} scores over {len(scores)} bands "
          f"(PLS n_components={args.n_components}); top {args.top}:")
    for rank, idx in enumerate(order, 1):
        pos = "" if positions is None else f"  @ {positions[rank - 1]:.6g}"
        print(f"  {rank:>2}. band[{idx}] = {scores[idx]:.4f}{pos}")
    if args.method == "vip":
        n_sel = int((scores > args.threshold).sum())
        print(f"\n{n_sel} / {len(scores)} bands exceed VIP threshold {args.threshold}.")
    if args.plot:
        _plot(scores, x_axis, args.method, args.threshold, args.plot)


if __name__ == "__main__":
    main()
