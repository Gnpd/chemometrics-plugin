#!/usr/bin/env python
"""Fit a preprocessing pipeline and transform spectra — without data leakage.

Fits the pipeline on the *training* data only (``--train`` if given, otherwise
``--input``), then transforms ``--input`` with the already-fitted pipeline. This
is the discipline that prevents leakage: scaling/scatter/OSC parameters are
learned from training spectra alone and merely applied to everything else.

Persists the fitted pipeline (joblib, or portable OpenModels JSON when ``--save-model``
ends in ``.json``) so the exact same transform can be replayed at prediction time.

Usage::

    # fit and transform the same set (exploration)
    python preprocess.py --input spectra.csv --spec spec.yaml --out processed.csv

    # fit on train, apply to test (deployment-correct)
    python preprocess.py --train train.csv --input test.csv --spec spec.yaml \
        --out test_processed.csv --save-model pipeline.joblib
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pipeline import build_pipeline, load_spec  # noqa: E402
from load_spectra import load_spectra  # noqa: E402


def _save(path: str, X: np.ndarray, x_axis: np.ndarray | None) -> None:
    if path.endswith(".npy"):
        np.save(path, X)
        return
    import pandas as pd

    columns = x_axis if (x_axis is not None and x_axis.shape[0] == X.shape[1]) else None
    df = pd.DataFrame(X, columns=columns)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Spectra to transform.")
    parser.add_argument("--spec", required=True, help="Pipeline spec (YAML/JSON).")
    parser.add_argument("--train", help="Fit on these spectra instead of --input.")
    parser.add_argument("--out", help="Write transformed spectra here (.csv/.parquet/.npy).")
    parser.add_argument("--save-model", help="Persist the fitted pipeline (.joblib or .json).")
    args = parser.parse_args()

    X, x_axis = load_spectra(args.input)
    if args.train:
        X_fit, x_axis_fit = load_spectra(args.train)
        x_axis = x_axis if x_axis is not None else x_axis_fit
    else:
        X_fit = X

    pipeline = build_pipeline(load_spec(args.spec), x_axis=x_axis)
    pipeline.fit(X_fit)
    X_out = pipeline.transform(X)

    print(f"fitted on {X_fit.shape[0]} spectra; transformed {X.shape} -> {X_out.shape}")
    n_bad = int(np.isnan(X_out).sum() + np.isinf(X_out).sum())
    if n_bad:
        print(f"WARNING: {n_bad} non-finite values in output (check fill values / order).")

    if args.out:
        # feature count may change (e.g. range_cut); only reuse x_axis if it matches
        _save(args.out, X_out, x_axis)
        print(f"wrote {args.out}")
    if args.save_model:
        if str(args.save_model).lower().endswith(".json"):
            import importlib.util
            from pathlib import Path

            persist_path = (
                Path(__file__).resolve().parents[2]
                / "model-persistence" / "scripts" / "persist_model.py"
            )
            spec = importlib.util.spec_from_file_location("_pp_persist_model", persist_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.save_model(pipeline, args.save_model, fmt="json")
        else:
            import joblib

            joblib.dump(pipeline, args.save_model)
        print(f"saved fitted pipeline -> {args.save_model}")


if __name__ == "__main__":
    main()
