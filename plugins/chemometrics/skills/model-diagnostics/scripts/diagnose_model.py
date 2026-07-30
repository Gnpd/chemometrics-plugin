#!/usr/bin/env python
"""Visual diagnostics for a fitted PCA/PLS model via the chemotools inspector.

Builds a ``PCAInspector`` or ``PLSRegressionInspector`` and emits the standard
diagnostic figures:

* PCA  — latent scores (with T²/Q limits), loadings, explained variance.
* PLS  — predicted-vs-actual, residuals, residual distribution/QQ, latent scores.

The inspector module is **experimental** (it emits a ``FutureWarning`` and its
API may change). This script suppresses that warning, degrades gracefully if the
inspector or matplotlib is unavailable, and always prints a text
``summary()`` so core numbers survive even when plotting does not. For the
prediction-time go/no-go check use ``applicability_domain.py`` instead — the
inspector is for *looking at* a model, not gating predictions.

Importable (:func:`build_inspector`, :func:`text_summary`, :func:`save_figures`)
and CLI-runnable::

    python diagnose_model.py --task regression --dataset fermentation --n-components 5 --outdir figs/
    python diagnose_model.py --task pca --dataset coffee --n-components 3 --no-plots
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np

# Figure-producing methods per inspector type (best-effort; skipped if absent).
_FIGURE_METHODS = {
    "regression": [
        "create_predicted_vs_actual_plot",
        "create_residuals_plot",
        "create_residual_distribution_plot",
        "create_latent_scores_figures",
    ],
    "pca": [
        "create_latent_scores_figures",
        "create_latent_loadings_figure",
        "create_latent_variance_figure",
    ],
}


def build_inspector(task: str, model, X_train, y_train=None, x_axis=None, confidence: float = 0.95):
    """Construct the appropriate inspector, suppressing its experimental warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if task == "regression":
            from chemotools.inspector import PLSRegressionInspector

            return PLSRegressionInspector(
                model, X_train=X_train, y_train=y_train, x_axis=x_axis, confidence=confidence
            )
        from chemotools.inspector import PCAInspector

        return PCAInspector(model, X_train=X_train, x_axis=x_axis, confidence=confidence)


def text_summary(inspector) -> str:
    """A plotting-free digest that always works (core numbers)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lines = [f"n_components : {inspector.n_components}",
                 f"n_samples    : {inspector.n_samples}",
                 f"n_features   : {inspector.n_features}"]
        for attr, label in [("R2_train", "R2 (train)"), ("RMSE_train", "RMSE (train)")]:
            if hasattr(inspector, attr):
                try:
                    lines.append(f"{label:<12} : {float(getattr(inspector, attr)):.4g}")
                except Exception:  # noqa: BLE001
                    pass
        lines.append(f"Hotelling T2 limit : {inspector.hotelling_t2_limit:.4g}")
        lines.append(f"Q-residuals limit  : {inspector.q_residuals_limit:.4g}")
    return "\n".join(lines)


def save_figures(inspector, task: str, outdir: str) -> list[str]:
    """Call each available figure method and save the results. Returns saved paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("(matplotlib not installed — skipping figures; pip install matplotlib)")
        return []
    Path(outdir).mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for method_name in _FIGURE_METHODS[task]:
            method = getattr(inspector, method_name, None)
            if method is None:
                continue
            try:
                result = method()
            except Exception as exc:  # noqa: BLE001  (experimental API)
                print(f"  (skipped {method_name}: {type(exc).__name__})")
                continue
            saved.extend(_save_figure_result(result, method_name, outdir))
    return saved


def _save_figure_result(result, base: str, outdir: str) -> list[str]:
    """A figure method may return a Figure or a dict/list of them."""
    figs: list[tuple[str, object]] = []
    if isinstance(result, dict):
        figs = [(f"{base}__{k}", v) for k, v in result.items()]
    elif isinstance(result, (list, tuple)):
        figs = [(f"{base}__{i}", v) for i, v in enumerate(result)]
    else:
        figs = [(base, result)]
    paths = []
    for name, fig in figs:
        if not hasattr(fig, "savefig"):
            continue
        path = str(Path(outdir) / f"{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        paths.append(path)
    return paths


def _modeling_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "chemometric-modeling" / "scripts" / "fit_model.py"
    )
    spec = importlib.util.spec_from_file_location("_md_fit_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["regression", "pca"])
    parser.add_argument("--dataset", choices=["fermentation", "coffee"], required=True)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--outdir", default="diagnostics_figs")
    parser.add_argument("--no-plots", action="store_true", help="Text summary only.")
    args = parser.parse_args()

    fm = _modeling_module()
    X, y, x_axis = fm.load_dataset(args.dataset)

    if args.task == "regression":
        model = fm.build_model("regression", n_components=args.n_components)
        model.fit(X, y)
    else:
        from sklearn.decomposition import PCA

        model = PCA(n_components=min(args.n_components, X.shape[0] - 1)).fit(X)
        y = None

    inspector = build_inspector(args.task, model, X, y, x_axis, args.confidence)
    print(text_summary(inspector))
    if not args.no_plots:
        saved = save_figures(inspector, args.task, args.outdir)
        print(f"\nsaved {len(saved)} figure(s) to {args.outdir}/")
        for p in saved:
            print("  ", p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(inspector, "close_figures"):
            inspector.close_figures()


if __name__ == "__main__":
    main()
