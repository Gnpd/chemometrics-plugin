#!/usr/bin/env python
"""Expand a spectral TRAINING set with synthetic augmented copies.

Applies a chain of chemotools augmentation transformers to the training spectra
several times to produce distinct augmented batches, then stacks the original
plus the augmented copies into one larger training set (labels tiled to match).

Each augmentation transformer redraws its random parameters on every
``transform`` call, so N transform calls give N different batches. A fixed
``--seed`` makes the whole expansion reproducible (two runs with the same seed
and spec produce identical output).

IMPORTANT: augment the **training** set only — never validation/test data (see
references/augmentation_workflow.md).

Spec format (YAML/JSON), same style as the hub's build_pipeline:

    - type: add_noise
      params: {distribution: gaussian, scale: 0.01}
    - type: index_shift
      params: {shift: 3}
    - type: spectrum_scale
      params: {scale: 0.05}

Usage::

    python augment.py --input train.csv --labels train_y.csv \
        --spec aug_spec.yaml --copies 5 --seed 0 \
        --out train_aug.csv --out-labels train_aug_y.csv
"""

from __future__ import annotations

import argparse
import importlib
import json

import numpy as np

REGISTRY = {
    "add_noise": ("chemotools.augmentation", "AddNoise"),
    "baseline_shift": ("chemotools.augmentation", "BaselineShift"),
    "spectrum_scale": ("chemotools.augmentation", "SpectrumScale"),
    "gaussian_broadening": ("chemotools.augmentation", "GaussianBroadening"),
    "index_shift": ("chemotools.augmentation", "IndexShift"),
    "fractional_shift": ("chemotools.augmentation", "FractionalShift"),
}


def _load_matrix(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        M = np.load(path)
    else:
        import pandas as pd

        M = pd.read_csv(path).to_numpy()
    return np.asarray(M, dtype=np.float64)


def load_spec(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def build_augmenter(spec: list[dict], seed: int | None):
    """A Pipeline of augmentation steps; seed each step for reproducibility."""
    from sklearn.pipeline import Pipeline

    steps = []
    for i, entry in enumerate(spec):
        module_name, class_name = REGISTRY[entry["type"]]
        cls = getattr(importlib.import_module(module_name), class_name)
        params = dict(entry.get("params") or {})
        # give each step a distinct but seed-derived random_state
        if seed is not None and "random_state" not in params:
            params["random_state"] = seed + i
        steps.append((f"{entry['type']}-{i}", cls(**params)))
    return Pipeline(steps)


def augment(X: np.ndarray, spec: list[dict], copies: int, seed: int | None) -> np.ndarray:
    """Return original stacked with `copies` augmented batches."""
    aug = build_augmenter(spec, seed)
    aug.fit(X)
    batches = [X]
    for _ in range(copies):
        batches.append(aug.transform(X))  # redraws each call
    return np.vstack(batches)


def _save_matrix(path: str, M: np.ndarray) -> None:
    if path.endswith(".npy"):
        np.save(path, M)
    else:
        import pandas as pd

        pd.DataFrame(M).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Training spectra (.csv/.npy).")
    parser.add_argument("--spec", required=True, help="Augmentation spec (YAML/JSON).")
    parser.add_argument("--labels", help="Training labels/reference values (.csv/.npy).")
    parser.add_argument("--copies", type=int, default=5, help="Augmented batches to add.")
    parser.add_argument("--seed", type=int, default=0, help="Reproducibility seed.")
    parser.add_argument("--out", help="Write expanded spectra.")
    parser.add_argument("--out-labels", help="Write tiled labels.")
    args = parser.parse_args()

    X = _load_matrix(args.input)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    spec = load_spec(args.spec)

    X_out = augment(X, spec, args.copies, args.seed)
    n_bad = int(np.isnan(X_out).sum() + np.isinf(X_out).sum())
    print(
        f"{X.shape} + {args.copies} copies -> {X_out.shape} "
        f"(x{args.copies + 1}); non-finite={n_bad}"
    )
    if n_bad:
        print("WARNING: non-finite values — check padding_mode / shift magnitude.")

    if args.out:
        _save_matrix(args.out, X_out)
        print(f"wrote {args.out}")
    if args.labels:
        y = _load_matrix(args.labels)
        y_out = np.vstack([y] * (args.copies + 1)) if y.ndim > 1 else np.tile(y, args.copies + 1)
        if args.out_labels:
            _save_matrix(args.out_labels, y_out)
            print(f"wrote {args.out_labels}  (labels tiled to {y_out.shape})")


if __name__ == "__main__":
    main()
