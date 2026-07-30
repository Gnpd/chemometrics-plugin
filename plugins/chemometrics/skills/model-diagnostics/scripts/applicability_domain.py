#!/usr/bin/env python
"""Applicability domain (AD): is a new spectrum safe to predict?

A model only extrapolates safely inside the region of spectral space it was
trained on. chemotools' ``outliers`` module (no sklearn equivalent) defines that
region from a fitted PCA/PLS model and flags new samples that fall outside it:

* **Hotelling T²** — distance *within* the model plane (unusual scores; an
  extreme-but-in-model sample).
* **Q-residuals** — distance *off* the model plane (spectral features the model
  cannot represent; a genuinely new kind of sample).

Both limits are **fit on the training model** at a chosen ``confidence`` and then
applied to new spectra. A sample outside either limit should not be trusted for
prediction — flag it, don't silently score it. See
``references/applicability_domain.md`` for T² vs Q vs leverage/DModX.

Importable (:func:`fit_domain`, :func:`flag`) and CLI-runnable — fit AD on a
training set, flag a test set::

    python applicability_domain.py --dataset fermentation --n-components 5 --confidence 0.95
    python applicability_domain.py --input X_train.csv --test X_new.csv --n-components 5
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

METHODS = ("hotelling_t2", "q_residuals")


def build_ad_model(X_train, n_components: int):
    """A PCA model of the training spectra — the reference space for the AD."""
    from sklearn.decomposition import PCA

    n = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    return PCA(n_components=n).fit(X_train)


def fit_domain(model, X_train, *, confidence: float = 0.95, methods=METHODS) -> dict:
    """Fit the requested outlier detectors on a fitted *model* + training X.

    Returns ``{method: fitted_detector}`` — each has a ``critical_value_`` and a
    ``predict`` (1 = inside domain, -1 = outside) / ``score_samples``.
    """
    from chemotools.outliers import HotellingT2, QResiduals

    registry = {"hotelling_t2": HotellingT2, "q_residuals": QResiduals}
    detectors = {}
    for name in methods:
        detectors[name] = registry[name](model, confidence=confidence).fit(X_train)
    return detectors


def flag(detectors: dict, X_new) -> dict:
    """Flag each sample in *X_new* against every detector.

    Returns ``{"per_method": {method: bool mask outside}, "outside_any": mask,
    "scores": {method: statistic}, "limits": {method: critical value}}``.
    ``True`` in a mask means **outside** the applicability domain.
    """
    per_method, scores, limits = {}, {}, {}
    for name, det in detectors.items():
        outside = det.predict(X_new) == -1  # sklearn convention: -1 = outlier
        per_method[name] = outside
        scores[name] = np.asarray(det.score_samples(X_new), dtype=float)
        limits[name] = float(det.critical_value_)
    outside_any = np.zeros(X_new.shape[0], dtype=bool)
    for mask in per_method.values():
        outside_any |= mask
    return {"per_method": per_method, "outside_any": outside_any,
            "scores": scores, "limits": limits}


def _modeling_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "chemometric-modeling" / "scripts" / "fit_model.py"
    )
    spec = importlib.util.spec_from_file_location("_md_fit_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sibling_loader():
    path = (
        Path(__file__).resolve().parents[2]
        / "spectral-preprocessing" / "scripts" / "load_spectra.py"
    )
    spec = importlib.util.spec_from_file_location("_md_load_spectra", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_projectable(model, X_probe) -> None:
    """The AD projects spectra with ``model.transform``; fail with guidance if it can't.

    JSON (OpenModels) serialization can drop *transform-only* fitted state — e.g. a
    ``PLSRegression`` restored from JSON predicts fine but currently lacks ``_x_std`` — so a
    transform-based diagnostic would otherwise crash deep inside scikit-learn.
    """
    try:
        model.transform(np.asarray(X_probe)[:1])
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"The applicability domain projects spectra with the model's transform(), which "
            f"failed on the loaded model ({type(exc).__name__}: {exc}). JSON serialization can "
            f"drop transform-only fitted state (e.g. PLS scaling), and that state cannot be "
            f"recovered from the JSON afterwards. For transform-based diagnostics, persist the "
            f"model as joblib at fit time (e.g. fit_model.py --out model.joblib) and pass that; "
            f"predict-based use like metrics still works from JSON."
        ) from exc


def _load_saved_model(path: str):
    """Load a persisted pipeline; ``.json`` via the model-persistence sibling, else joblib."""
    if str(path).lower().endswith(".json"):
        persist_path = (
            Path(__file__).resolve().parents[2]
            / "model-persistence" / "scripts" / "persist_model.py"
        )
        spec = importlib.util.spec_from_file_location("_md_persist_model", persist_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_model(path)
    import joblib

    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["fermentation", "coffee"])
    src.add_argument("--input", help="Training spectra.")
    parser.add_argument("--test", help="New spectra to flag (default: score the training set).")
    parser.add_argument("--model", help="Fitted model/pipeline (.joblib or .json) to define the "
                                        "AD; default builds a PCA of the training spectra.")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.dataset:
        X_train, _, _ = _modeling_module().load_dataset(args.dataset)
    else:
        X_train, _ = _sibling_loader().load_spectra(args.input)
    X_new = X_train
    if args.test:
        X_new, _ = _sibling_loader().load_spectra(args.test)

    if args.model:
        model = _load_saved_model(args.model)
        _require_projectable(model, X_train)
    else:
        model = build_ad_model(X_train, args.n_components)

    detectors = fit_domain(model, X_train, confidence=args.confidence)
    result = flag(detectors, X_new)

    print(f"Applicability domain (confidence={args.confidence}) on {X_new.shape[0]} samples:")
    for name, det in detectors.items():
        n_out = int(result["per_method"][name].sum())
        print(f"  {name:<13} limit={result['limits'][name]:.4g}  outside={n_out}")
    outside_idx = np.where(result["outside_any"])[0]
    print(f"\noutside the domain (any test): {len(outside_idx)} / {X_new.shape[0]}")
    if len(outside_idx):
        print("  sample indices:", outside_idx.tolist()[:50])
        print("  -> do NOT trust predictions for these without re-calibration.")


if __name__ == "__main__":
    main()
