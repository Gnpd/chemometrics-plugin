---
name: spectral-preprocessing
description: >-
  Preprocess vibrational spectroscopy data (IR, NIR, Raman, ATR-FTIR, UV-Vis) with
  chemotools and scikit-learn. Use when the user wants to clean, correct, normalize,
  smooth, differentiate, or build a preprocessing pipeline for spectra before PCA/PLS
  modeling — including choosing which corrections to apply and in what order, assembling
  a leakage-free scikit-learn Pipeline, and diagnosing the result. Delegates deep
  technique choices to the baseline-scatter-correction, smoothing-derivatives, and
  calibration-transfer skills.
---

# Spectral preprocessing with chemotools

Turn raw spectra into model-ready features by assembling a `scikit-learn`
`Pipeline` of `chemotools` transformers, in the right order, fit without data
leakage. This skill orchestrates the workflow and routes hard technique choices
to the specialized skills.

## Prerequisite

`pip install chemotools` (add `chemotools[viz]` for plotting, `pandas` for
CSV/dataset loading). The scripts in `scripts/` need only an installed
chemotools plus numpy/scipy/scikit-learn; pandas/matplotlib/pyyaml/joblib are
used by some and degrade or error with an actionable message when absent.

## Workflow

### 1. Understand the data
Establish, by asking or inspecting:
- **Modality** (IR / NIR / Raman / ATR-FTIR / UV-Vis) — drives the recipe.
- **Shape** — must be `(n_samples, n_features)`; reshape a single spectrum to
  `(1, -1)`. Use `scripts/load_spectra.py` to load and report shape, x-axis, and
  non-finite values.
- **X-axis** — is a wavenumber/wavelength array available (often the CSV header)?
  Needed for region cuts and band scalers in physical units.
- **Split & task** — is there a train/test split? Regression, classification, or
  exploration? This fixes where scaling and supervised steps go.

### 2. Decide the steps
Follow the canonical order (`references/ordering_guide.md`):
`units → region cut → baseline → scatter → smoothing → derivative → scaling → model`.
Use `assets/decision_tree.svg` and `references/technique_catalog.md` to pick
transformers. For the genuinely hard calls, **read the relevant spoke** (below).
Prefer the **minimal** set of corrections — over-processing is a common failure.

### 3. Assemble the pipeline
Write a spec (start from `assets/pipeline_spec.yaml`) and build it with
`scripts/build_pipeline.py`, or hand-write a `Pipeline` from
`assets/pipeline_template.py`. The `$x_axis` sentinel in a spec is replaced with
the loaded axis. Keep **every** step in one Pipeline so fitting is leakage-free.

### 4. Fit and transform
`scripts/preprocess.py` fits on training data only (`--train`), transforms the
input, and persists the fitted pipeline (`--save-model`). Never `fit_transform`
the test set. See `references/pitfalls.md` §1.

### 5. Verify
- `scripts/diagnose.py` — per-step shape/stats and red flags (NaN, all-zero,
  constant rows). Run this whenever a result looks wrong.
- `scripts/plot_spectra.py` — raw vs. processed overlay.
Check that peaks are preserved, no unintended NaNs appeared, and the region/units
are what you expect.

## When to reach for a spoke

| The hard question | Skill to read |
|---|---|
| Which baseline (AirPLS/ArPLS/AsLS/polynomial/rubberband) and what `lam`? Which scatter correction (SNV/RNV/MSC/EMSC)? | **baseline-scatter-correction** |
| What smoothing window/order? Which derivative order? Balancing noise vs. resolution? | **smoothing-derivatives** |
| Spectra on different x-axis grids, multi-instrument transfer, per-sample background/metadata at transform time? | **calibration-transfer** |
| Synthetic data augmentation for robust model training? | **spectral-augmentation** (if installed) |

## Scripts

| script | purpose |
|---|---|
| `scripts/load_spectra.py` | Load CSV/Parquet/npy → `(X, x_axis)`; reshape 1-D; quality report. |
| `scripts/build_pipeline.py` | Registry + spec→`Pipeline` builder (importable; used by the others). |
| `scripts/preprocess.py` | Fit-on-train / transform / persist (leakage-free CLI). |
| `scripts/diagnose.py` | Per-step cumulative diagnosis with red-flag detection. |
| `scripts/plot_spectra.py` | Raw vs. preprocessed overlay (matplotlib optional). |

## References

- `references/technique_catalog.md` — every transformer, params, when to use.
- `references/ordering_guide.md` — the canonical order + named recipes per modality.
- `references/pitfalls.md` — leakage, shapes, x-axis, NaNs, over-processing.
- `references/api_contract.md` — transformer/Pipeline API, current param names, metadata routing, persistence, datasets.
