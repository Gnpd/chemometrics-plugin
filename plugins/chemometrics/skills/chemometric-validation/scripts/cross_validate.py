#!/usr/bin/env python
"""Cross-validate a full spectra->property pipeline without leakage.

Runs ``sklearn.model_selection.cross_validate`` on the *whole* pipeline
(preprocessing + model together), so every fold refits preprocessing on its own
training rows — the only leakage-free way to CV a chemometric model.

Fold strategy:

* **grouped** (``--groups``) — ``GroupKFold`` so replicate or augmented spectra
  of one physical sample never straddle a fold. Use this whenever such copies
  exist; otherwise CV is optimistically biased. This is the discipline the
  spectral-augmentation skill's copies require.
* **stratified** — ``StratifiedKFold`` for classification (keeps class balance).
* **plain**      — ``KFold`` for regression without groups.

Reports each metric as ``mean ± std`` across folds (the spread matters as much as
the mean). Importable (:func:`run_cv`) and CLI-runnable::

    python cross_validate.py --task regression --dataset fermentation --n-components 6 --cv 5
    python cross_validate.py --task classification --dataset coffee --n-components 4
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _modeling_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "chemometric-modeling"
        / "scripts"
        / "fit_model.py"
    )
    spec = importlib.util.spec_from_file_location("_cv_fit_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scorers(task: str) -> dict:
    from sklearn.metrics import make_scorer

    if task == "regression":
        def _rpd(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float).ravel()
            rmse = np.sqrt(np.mean((np.asarray(y_pred, float).ravel() - y_true) ** 2))
            std = np.std(y_true, ddof=1) if len(y_true) > 1 else 0.0
            return std / rmse if rmse > 0 else float("inf")

        def _bias(y_true, y_pred):
            return float(np.mean(np.asarray(y_pred, float).ravel()
                                 - np.asarray(y_true, float).ravel()))

        return {
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "rpd": make_scorer(_rpd, greater_is_better=True),
            "bias": make_scorer(_bias, greater_is_better=True),
        }
    return {"accuracy": "accuracy", "f1_macro": "f1_macro"}


def make_splitter(task: str, cv: int, groups):
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    if groups is not None:
        return GroupKFold(n_splits=cv)
    if task == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    return KFold(n_splits=cv, shuffle=True, random_state=0)


def run_cv(pipe, X, y, *, task: str, cv: int = 5, groups=None) -> dict:
    """Cross-validate *pipe* and return ``{metric: {"mean":..., "std":..., "folds":[...]}}``.

    RMSE is reported as a positive number (sklearn scores it as neg RMSE).
    """
    from sklearn.model_selection import cross_validate

    splitter = make_splitter(task, cv, groups)
    scoring = _scorers(task)
    res = cross_validate(
        pipe, X, y, cv=splitter, groups=groups, scoring=scoring, return_train_score=False
    )
    out: dict[str, dict] = {}
    for metric in scoring:
        folds = np.asarray(res[f"test_{metric}"], dtype=float)
        if metric == "rmse":  # stored as negative
            folds = -folds
        out[metric] = {
            "mean": float(folds.mean()),
            "std": float(folds.std(ddof=1)) if len(folds) > 1 else 0.0,
            "folds": folds.tolist(),
        }
    return out


def format_report(results: dict) -> str:
    lines = [f"{'metric':<10} {'mean':>10} {'±std':>10}"]
    for metric, d in results.items():
        lines.append(f"{metric:<10} {d['mean']:>10.4f} {d['std']:>10.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["regression", "classification"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["fermentation", "coffee"])
    src.add_argument("--input")
    parser.add_argument("--y")
    parser.add_argument("--groups", help="Group-label file (.npy/.csv) for GroupKFold.")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--spec", help="Preprocessing-head spec (YAML/JSON).")
    args = parser.parse_args()

    fm = _modeling_module()
    if args.dataset:
        X, y, x_axis = fm.load_dataset(args.dataset)
    else:
        X, y, x_axis = fm._load_xy(args)
    groups = np.load(args.groups) if args.groups and args.groups.endswith(".npy") else (
        np.loadtxt(args.groups) if args.groups else None
    )
    prep_spec = fm._load_build_pipeline().load_spec(args.spec) if args.spec else None

    pipe = fm.build_full_pipeline(
        args.task, n_components=args.n_components, prep_spec=prep_spec, x_axis=x_axis
    )
    results = run_cv(pipe, X, y, task=args.task, cv=args.cv, groups=groups)
    strategy = "GroupKFold" if groups is not None else (
        "StratifiedKFold" if args.task == "classification" else "KFold"
    )
    print(f"{args.cv}-fold {strategy} on X{X.shape} (n_components={args.n_components}):\n")
    print(format_report(results))


if __name__ == "__main__":
    main()
