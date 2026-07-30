#!/usr/bin/env python
"""Choose the number of PLS latent variables (LVs) by cross-validation.

Sweeps ``n_components`` and cross-validates the *whole* pipeline at each value,
then reports two optima:

* **min**  — the LV count with the best mean CV score (lowest RMSECV for
  regression, highest accuracy for classification);
* **1-SE** — the *most parsimonious* model within one standard error of the best
  (Breiman's one-standard-error rule) — the recommended choice, because it
  guards against over-fitting on the CV noise.

Selecting LVs by CV (never by R²/accuracy on the training data, which only ever
improves with more LVs) is the core anti-over-fit discipline of PLS. See
``references/component_selection.md``.

Use grouped folds (``--groups``) whenever replicate or augmented spectra of one
physical sample would otherwise straddle folds — see the chemometric-validation
skill.

Importable (:func:`sweep`, :func:`select`) and CLI-runnable::

    python select_components.py --task regression --dataset fermentation --max-components 15
    python select_components.py --task classification --dataset coffee --max-components 10 --plot lv.png
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


def _scoring(task: str) -> tuple[str, bool]:
    """Return (sklearn scoring name, higher_is_better)."""
    if task == "regression":
        return "neg_root_mean_squared_error", True  # neg RMSE: higher (less negative) is better
    return "accuracy", True


def sweep(
    task: str,
    X,
    y,
    *,
    max_components: int,
    prep_spec=None,
    x_axis=None,
    cv=5,
    groups=None,
    scale: bool = True,
):
    """Cross-validate the pipeline for ``n_components`` in ``1..max_components``.

    Returns a list of dicts: ``n_components``, ``mean``, ``se`` (standard error of
    the mean over folds), and ``metric`` (positive RMSECV or accuracy).
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score

    fm = _fit_model_module()
    scoring, _ = _scoring(task)

    if groups is not None:
        splitter = GroupKFold(n_splits=cv)
    elif task == "classification":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=0)

    n_max = min(max_components, X.shape[1], X.shape[0] - 1)
    rows = []
    for nc in range(1, n_max + 1):
        pipe = fm.build_full_pipeline(
            task, n_components=nc, scale=scale, prep_spec=prep_spec, x_axis=x_axis
        )
        scores = cross_val_score(
            pipe, X, y, cv=splitter, groups=groups, scoring=scoring
        )
        mean = float(scores.mean())
        se = float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
        metric = -mean if task == "regression" else mean  # report positive RMSECV
        rows.append({"n_components": nc, "mean": mean, "se": se, "metric": metric})
    return rows


def select(rows: list[dict]) -> dict:
    """Apply min and 1-SE rules to a sweep. Higher ``mean`` is always better here."""
    best = max(rows, key=lambda r: r["mean"])
    threshold = best["mean"] - best["se"]  # within 1 SE of the best mean
    parsimonious = min(
        (r for r in rows if r["mean"] >= threshold), key=lambda r: r["n_components"]
    )
    return {
        "min": best["n_components"],
        "one_se": parsimonious["n_components"],
        "best_metric": best["metric"],
        "one_se_metric": parsimonious["metric"],
    }


def _plot(rows, task, choice, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot; pip install matplotlib)")
        return
    xs = [r["n_components"] for r in rows]
    ys = [r["metric"] for r in rows]
    es = [r["se"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
    ax.axvline(choice["min"], color="tab:gray", ls="--", label=f"min ({choice['min']})")
    ax.axvline(choice["one_se"], color="tab:green", ls="-", label=f"1-SE ({choice['one_se']})")
    ax.set_xlabel("n_components (latent variables)")
    ax.set_ylabel("RMSECV" if task == "regression" else "CV accuracy")
    ax.set_title(f"LV selection — {task}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["regression", "classification"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["fermentation", "coffee"])
    src.add_argument("--input")
    parser.add_argument("--y")
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--spec", help="Preprocessing-head spec (YAML/JSON).")
    parser.add_argument("--plot", help="Write the RMSECV/accuracy-vs-LV curve here.")
    args = parser.parse_args()

    fm = _fit_model_module()
    if args.dataset:
        X, y, x_axis = fm.load_dataset(args.dataset)
    else:
        X, y, x_axis = fm._load_xy(args)
    prep_spec = fm._load_build_pipeline().load_spec(args.spec) if args.spec else None

    rows = sweep(
        args.task, X, y,
        max_components=args.max_components, prep_spec=prep_spec, x_axis=x_axis, cv=args.cv,
    )
    choice = select(rows)
    unit = "RMSECV" if args.task == "regression" else "accuracy"
    print(f"{'LV':>3}  {unit:>10}  {'±SE':>8}")
    for r in rows:
        print(f"{r['n_components']:>3}  {r['metric']:>10.4f}  {r['se']:>8.4f}")
    print(
        f"\nbest ({unit}) at {choice['min']} LV: {choice['best_metric']:.4f}\n"
        f"recommended (1-SE, parsimonious): {choice['one_se']} LV "
        f"({choice['one_se_metric']:.4f})"
    )
    if args.plot:
        _plot(rows, args.task, choice, args.plot)


if __name__ == "__main__":
    main()
