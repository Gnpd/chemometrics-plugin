#!/usr/bin/env python
"""Chemometric performance metrics on a held-out set.

Regression (the chemometric aggregates sklearn doesn't ship):

* **RMSEP** — root mean squared error of prediction (same units as y).
* **R²**    — coefficient of determination.
* **RPD**   — ratio of performance to deviation, ``std(y) / RMSEP``. A
  units-free quality index: >2 useful, >3 good, >5 excellent for many
  applications (see ``references/metrics_reference.md``).
* **bias**  — mean signed error ``mean(ŷ − y)``; a systematic offset survives
  even a low-variance model and flags a transfer/drift problem.
* **SEP**   — standard error of prediction (bias-corrected RMSEP).

Classification: accuracy, macro-F1, and the confusion matrix.

Only the chemometric aggregates are wrapped here; everything else defers to
``sklearn.metrics`` (the plugin does not re-implement sklearn). Importable
(:func:`regression_metrics`, :func:`classification_metrics`) and CLI-runnable on
a saved model + dataset::

    python metrics.py --task regression --dataset fermentation --model pls.joblib
"""

from __future__ import annotations

import argparse

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSEP, R², RPD, bias, SEP for a continuous target."""
    from sklearn.metrics import mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rmsep = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    residuals = y_pred - y_true
    # SEP: spread of the residuals about their mean (bias removed)
    sep = float(np.sqrt(np.sum((residuals - bias) ** 2) / max(len(y_true) - 1, 1)))
    std_y = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    rpd = float(std_y / rmsep) if rmsep > 0 else float("inf")
    return {
        "rmsep": rmsep,
        "r2": float(r2_score(y_true, y_pred)),
        "rpd": rpd,
        "bias": bias,
        "sep": sep,
        "n": int(len(y_true)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, labels, and the confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(np.unique(np.concatenate([y_true, y_pred])))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def format_report(metrics: dict, task: str) -> str:
    if task == "regression":
        return (
            f"n     : {metrics['n']}\n"
            f"RMSEP : {metrics['rmsep']:.4g}\n"
            f"R^2   : {metrics['r2']:.4f}\n"
            f"RPD   : {metrics['rpd']:.3f}\n"
            f"bias  : {metrics['bias']:+.4g}\n"
            f"SEP   : {metrics['sep']:.4g}"
        )
    lines = [
        f"n        : {metrics['n']}",
        f"accuracy : {metrics['accuracy']:.4f}",
        f"F1 macro : {metrics['f1_macro']:.4f}",
        f"labels   : {metrics['labels']}",
        "confusion:",
    ]
    for row in metrics["confusion_matrix"]:
        lines.append("  " + "  ".join(f"{v:>4d}" for v in row))
    return "\n".join(lines)


def _modeling_module():
    """Reuse the modeling skill's dataset loader (portable sibling import)."""
    import importlib.util
    from pathlib import Path

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


def _load_saved_model(path: str):
    """Load a persisted pipeline; ``.json`` via the model-persistence sibling, else joblib."""
    if str(path).lower().endswith(".json"):
        import importlib.util
        from pathlib import Path

        persist_path = (
            Path(__file__).resolve().parents[2]
            / "model-persistence" / "scripts" / "persist_model.py"
        )
        spec = importlib.util.spec_from_file_location("_cv_persist_model", persist_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_model(path)
    import joblib

    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["regression", "classification"])
    parser.add_argument("--model", required=True, help="Fitted pipeline (.joblib or .json).")
    parser.add_argument("--dataset", choices=["fermentation", "coffee"])
    parser.add_argument("--input", help="Held-out spectra (.csv/.parquet/.npy).")
    parser.add_argument("--y", help="Held-out target.")
    args = parser.parse_args()

    try:
        model = _load_saved_model(args.model)
    except ImportError:
        parser.error("Loading a saved model needs joblib: pip install joblib.")

    fm = _modeling_module()
    if args.dataset:
        X, y, _ = fm.load_dataset(args.dataset)
    elif args.input and args.y:
        X, y, _ = fm._load_xy(args)
    else:
        parser.error("Provide --dataset, or both --input and --y.")

    y_pred = model.predict(X)
    if args.task == "regression":
        report = regression_metrics(y, y_pred)
    else:
        report = classification_metrics(y, y_pred)
    print(format_report(report, args.task))


if __name__ == "__main__":
    main()
