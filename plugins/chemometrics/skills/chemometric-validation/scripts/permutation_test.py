#!/usr/bin/env python
"""Permutation (y-scramble) significance test for a chemometric model.

Is the model's cross-validated performance better than chance? Refit the whole
pipeline many times on **shuffled** targets to build the null distribution of
the CV score, then compare the true score against it. The p-value is the
fraction of permutations that match or beat the real model.

This is the standard guard against **chance correlation** — with spectra (many
more bands than samples) a flexible model can fit random labels, so "high R² /
accuracy" alone proves nothing. A significant permutation test does.

Wraps ``sklearn.model_selection.permutation_test_score`` (which refits per
permutation, no leakage) and honours grouped folds. Importable
(:func:`permutation_test`) and CLI-runnable::

    python permutation_test.py --task regression --dataset fermentation --n-components 6 --n-permutations 200
    python permutation_test.py --task classification --dataset coffee --n-components 4
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
    spec = importlib.util.spec_from_file_location("_pt_fit_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cv_module():
    path = Path(__file__).resolve().parent / "cross_validate.py"
    spec = importlib.util.spec_from_file_location("_pt_cross_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def permutation_test(
    pipe, X, y, *, task: str, cv: int = 5, groups=None, n_permutations: int = 200, random_state: int = 0
) -> dict:
    """Return ``{score, permutation_scores, p_value, scoring}``.

    Higher score is always better here: R² for regression, accuracy for
    classification. A small p-value means the true model beats scrambled-label
    models — evidence the signal is real.
    """
    from sklearn.model_selection import permutation_test_score

    splitter = _cv_module().make_splitter(task, cv, groups)
    scoring = "r2" if task == "regression" else "accuracy"
    score, perm_scores, p_value = permutation_test_score(
        pipe, X, y,
        groups=groups, cv=splitter, scoring=scoring,
        n_permutations=n_permutations, random_state=random_state,
    )
    return {
        "score": float(score),
        "permutation_scores": np.asarray(perm_scores, dtype=float),
        "p_value": float(p_value),
        "scoring": scoring,
    }


def _plot(result, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot; pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(result["permutation_scores"], bins=20, color="tab:gray", alpha=0.7,
            label="permuted (null)")
    ax.axvline(result["score"], color="tab:green", lw=2,
               label=f"true = {result['score']:.3f}")
    ax.set_xlabel(f"{result['scoring']} under label permutation")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation test  (p = {result['p_value']:.4f})")
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
    parser.add_argument("--groups")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--spec", help="Preprocessing-head spec (YAML/JSON).")
    parser.add_argument("--plot")
    args = parser.parse_args()

    fm = _modeling_module()
    if args.dataset:
        X, y, x_axis = fm.load_dataset(args.dataset)
    else:
        X, y, x_axis = fm._load_xy(args)
    groups = (
        np.load(args.groups) if args.groups and args.groups.endswith(".npy")
        else (np.loadtxt(args.groups) if args.groups else None)
    )
    prep_spec = fm._load_build_pipeline().load_spec(args.spec) if args.spec else None

    pipe = fm.build_full_pipeline(
        args.task, n_components=args.n_components, prep_spec=prep_spec, x_axis=x_axis
    )
    result = permutation_test(
        pipe, X, y, task=args.task, cv=args.cv, groups=groups,
        n_permutations=args.n_permutations,
    )
    print(
        f"true {result['scoring']} = {result['score']:.4f}\n"
        f"null mean {result['scoring']} = {result['permutation_scores'].mean():.4f} "
        f"(over {len(result['permutation_scores'])} permutations)\n"
        f"p-value = {result['p_value']:.4f}  "
        f"({'significant' if result['p_value'] < 0.05 else 'NOT significant'} at 0.05)"
    )
    if args.plot:
        _plot(result, args.plot)


if __name__ == "__main__":
    main()
