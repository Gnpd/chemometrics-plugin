#!/usr/bin/env python
"""Build and fit a chemometric model — preprocessing + estimator in one Pipeline.

Three tasks:

* ``regression``     — PLS regression (chemotools ``PLSRegression``) for a
  continuous property (glucose, moisture, octane, ...).
* ``classification`` — PLS-DA via the :class:`PLSDA` wrapper below (PLS on a
  one-hot target, predict by ``argmax``). See ``references/modeling_methods.md``
  for the alternative (LDA on PLS scores).
* ``pca``            — PCA (sklearn) for unsupervised exploration; no ``y``.

The model can be prefixed with a preprocessing head built from the
``spectral-preprocessing`` hub's spec format (``--spec spec.yaml``), so the whole
chain — preprocess → model — is a single scikit-learn ``Pipeline`` that is
**fit on train only**. That is the object you cross-validate and persist.

Importable (the smoke test calls :func:`build_model` / :func:`build_full_pipeline`)
and CLI-runnable::

    # fit PLS on a bundled dataset and persist the pipeline
    python fit_model.py --task regression --dataset fermentation --n-components 6 --out pls.joblib

    # PLS-DA on your own CSVs with an SNV + Savitzky-Golay preprocessing head
    python fit_model.py --task classification --input X.csv --y y.csv \
        --spec prep.yaml --n-components 4 --out plsda.joblib
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS-DA classifier: PLS regression on one-hot classes, predict by argmax.

    The plugin's documented default classification recipe. It is a genuine
    scikit-learn classifier (``fit``/``predict``/``predict_proba``) so it drops
    into a ``Pipeline`` and ``cross_val_score`` unchanged. For the two-stage
    alternative (LDA on PLS scores) see ``references/modeling_methods.md``.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS latent variables.
    scale : bool, default=True
        Passed through to ``PLSRegression`` (scale each column to unit variance).
    """

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        from chemotools.regression import PLSRegression

        self._lb = LabelBinarizer()
        Y = self._lb.fit_transform(y)
        if Y.shape[1] == 1:  # binary -> two columns so argmax is well defined
            Y = np.hstack([1 - Y, Y])
        self.classes_ = self._lb.classes_
        self.pls_ = PLSRegression(n_components=self.n_components, scale=self.scale)
        self.pls_.fit(X, Y)
        return self

    def decision_function(self, X):
        return np.asarray(self.pls_.predict(X))

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Softmax over the PLS scores — a convenience, not a calibrated probability."""
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


def build_model(task: str, *, n_components: int = 2, scale: bool = True) -> BaseEstimator:
    """Return the bare estimator for *task* (no preprocessing head).

    task : ``"regression"`` | ``"classification"`` | ``"pca"``.
    """
    if task == "regression":
        from chemotools.regression import PLSRegression

        return PLSRegression(n_components=n_components, scale=scale)
    if task == "classification":
        return PLSDA(n_components=n_components, scale=scale)
    if task == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components)
    raise ValueError(f"Unknown task {task!r}. Use regression | classification | pca.")


def _load_build_pipeline():
    """Import the hub's ``build_pipeline`` by relative path (portable, no plugin paths)."""
    hub = (
        Path(__file__).resolve().parents[2]
        / "spectral-preprocessing"
        / "scripts"
        / "build_pipeline.py"
    )
    if not hub.exists():
        raise FileNotFoundError(
            "A --spec preprocessing head needs the spectral-preprocessing skill "
            f"alongside this one (looked for {hub}). Install the full plugin, or "
            "drop the --spec flag to fit the model on raw features."
        )
    spec = importlib.util.spec_from_file_location("_hub_build_pipeline", hub)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _persist_module():
    """Sibling-load the model-persistence skill (for JSON export)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "model-persistence"
        / "scripts"
        / "persist_model.py"
    )
    spec = importlib.util.spec_from_file_location("_persist_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out_format(path: str, fmt: str) -> str:
    """Resolve joblib | json from an explicit --format or the --out extension."""
    if fmt and fmt != "auto":
        return fmt
    return "json" if str(path).lower().endswith(".json") else "joblib"


def build_full_pipeline(
    task: str,
    *,
    n_components: int = 2,
    scale: bool = True,
    prep_spec: list[dict[str, Any]] | None = None,
    x_axis: Any = None,
):
    """Preprocessing head (optional) + model, as one ``Pipeline``.

    With ``prep_spec=None`` this is just ``Pipeline([(task, model)])`` so the
    return type is uniform and always cross-validatable / persistable.
    """
    from sklearn.pipeline import Pipeline

    steps: list[tuple[str, Any]] = []
    if prep_spec:
        head = _load_build_pipeline().build_pipeline(prep_spec, x_axis=x_axis)
        steps.extend(head.steps)
    model_name = {"regression": "pls", "classification": "plsda", "pca": "pca"}[task]
    steps.append((model_name, build_model(task, n_components=n_components, scale=scale)))
    return Pipeline(steps)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load a bundled chemotools dataset by short name.

    ``"fermentation"`` -> (X, glucose y, wavenumbers) regression;
    ``"coffee"``       -> (X, origin y, wavenumbers) 3-class PLS-DA.
    Returns ``(X, y, x_axis)`` with ``y=None`` for unsupervised use.
    """
    from chemotools import datasets

    if name == "fermentation":
        X, y = datasets.load_fermentation_train()
        return _df_xy(X, y)
    if name == "coffee":
        X, y = datasets.load_coffee()
        return _df_xy(X, y)
    raise ValueError(f"Unknown dataset {name!r}. Use fermentation | coffee.")


def _df_xy(X, y):
    x_axis = None
    try:
        x_axis = np.asarray([float(c) for c in X.columns], dtype=np.float64)
    except (TypeError, ValueError):
        x_axis = None
    return X.to_numpy(dtype=np.float64), y.to_numpy().ravel(), x_axis


def _load_xy(args):
    """Resolve (X, y, x_axis) from either --dataset or --input/--y."""
    if args.dataset:
        return load_dataset(args.dataset)
    ls = _sibling_loader()
    X, x_axis = ls.load_spectra(args.input)
    y = None
    if args.y:
        y_arr, _ = ls.load_spectra(args.y, header=False)
        y = y_arr.ravel()
    return X, y, x_axis


def _sibling_loader():
    path = (
        Path(__file__).resolve().parents[2]
        / "spectral-preprocessing"
        / "scripts"
        / "load_spectra.py"
    )
    spec = importlib.util.spec_from_file_location("_hub_load_spectra", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["regression", "classification", "pca"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["fermentation", "coffee"], help="Bundled dataset.")
    src.add_argument("--input", help="Spectra file (.csv/.parquet/.npy).")
    parser.add_argument("--y", help="Target file (needed with --input for supervised tasks).")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--no-scale", action="store_true", help="Disable PLS column scaling.")
    parser.add_argument("--spec", help="Preprocessing-head spec (YAML/JSON, hub format).")
    parser.add_argument("--out", help="Persist the fitted pipeline here (.joblib or .json).")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["auto", "joblib", "json"],
        default="auto",
        help="Persistence format (default: infer from the --out extension). "
        "json uses OpenModels (portable, pickle-free); see the model-persistence skill.",
    )
    args = parser.parse_args()

    X, y, x_axis = _load_xy(args)
    prep_spec = None
    if args.spec:
        prep_spec = _load_build_pipeline().load_spec(args.spec)

    pipe = build_full_pipeline(
        args.task,
        n_components=args.n_components,
        scale=not args.no_scale,
        prep_spec=prep_spec,
        x_axis=x_axis,
    )
    if args.task == "pca":
        pipe.fit(X)
    else:
        if y is None:
            parser.error(f"--task {args.task} needs a target (use --dataset or --y).")
        pipe.fit(X, y)

    print(f"Fitted pipeline ({args.task}, n_components={args.n_components}) on X{X.shape}:")
    print(pipe)
    if args.task == "classification":
        print(f"train accuracy : {(pipe.predict(X) == y).mean():.3f}")
    elif args.task == "regression":
        from sklearn.metrics import r2_score

        print(f"train R^2      : {r2_score(y, pipe.predict(X)):.3f}")
    else:
        evr = pipe.named_steps["pca"].explained_variance_ratio_
        print(f"explained var  : {np.cumsum(evr).round(3)[: min(5, len(evr))]} ...")

    if args.out:
        fmt = _out_format(args.out, args.fmt)
        if fmt == "json":
            _persist_module().save_model(pipe, args.out, fmt="json")
        else:
            try:
                import joblib
            except ImportError:
                parser.error("Persisting needs joblib: pip install joblib.")
            joblib.dump(pipe, args.out)
        print(f"saved -> {args.out} ({fmt})")


if __name__ == "__main__":
    main()
