---
name: chemometric-modeling
description: >-
  Build and fit chemometric models on spectra with scikit-learn and chemotools — PCA for
  exploration, PLS regression for quantitative prediction, and PLS-DA / classification for
  categorical outcomes. Use when choosing a model, selecting the number of PLS latent
  variables, wiring preprocessing + model into one scikit-learn Pipeline, or doing supervised
  band selection (VIP/SR). Delegates preprocessing to spectral-preprocessing and evaluation to
  chemometric-validation.
---

# Chemometric modeling

Turn preprocessed spectra into a fitted model: **PCA** to explore, **PLS
regression** to predict a continuous property, **PLS-DA** to predict a class.
This skill owns the *model* stage of the workflow; preprocessing belongs to
**spectral-preprocessing** and rigorous evaluation to **chemometric-validation**.

## Pick the model

Decide from the task (`assets/modeling_decision.svg`):

1. **Explore / no target yet** → **PCA** (sklearn). Scores reveal clusters and
   gross outliers; loadings show which bands drive the variance. Not a predictor.
2. **Predict a number** (glucose, moisture, octane) → **PLS regression**
   (chemotools `PLSRegression`).
3. **Predict a class** (origin, grade, pass/fail) → **PLS-DA** — PLS regression
   on a one-hot target, predict by `argmax` (the plugin default; the two-stage
   *LDA-on-PLS-scores* alternative is in `references/modeling_methods.md`).

PLS is the chemometrics workhorse because spectra are wide (many more bands than
samples) and highly collinear — it projects onto a few latent variables (LVs)
that covary with the target. Details in `references/modeling_methods.md`.

## One pipeline: preprocess → model

Always fit preprocessing **and** the model as a single `Pipeline`, so the whole
chain is **fit on train only** and cross-validates without leakage. The
preprocessing head reuses the hub's spec format:

```
# preprocess (SNV + 1st-derivative) then PLS, fit on a bundled dataset, persist
python scripts/fit_model.py --task regression --dataset fermentation \
    --spec prep.yaml --n-components 6 --out pls.joblib
```
Drop `--spec` to model raw features, or point `--input/--y` at your own data.
`build_model` / `build_full_pipeline` are importable for use in notebooks.

## Choose the number of latent variables — by CV, never by fit

More LVs always fit the training data better and eventually memorize noise.
Choose LVs by **cross-validation**: pick the parsimonious model within one
standard error of the best CV score (the **1-SE rule**).

```
python scripts/select_components.py --task regression --dataset fermentation \
    --max-components 15 --plot lv_curve.png
```
Reports both the `min`-RMSECV LV count and the recommended 1-SE choice, and plots
RMSECV (regression) or accuracy (classification) vs. LV. Rationale and the
over/under-fitting picture: `references/component_selection.md`. Use grouped
folds for replicate/augmented spectra — see **chemometric-validation**.

## Which bands matter — VIP / SR

From a fitted PLS, rank wavelengths by their contribution (chemometrics-specific;
no sklearn equivalent):

```
python scripts/feature_importance.py --dataset fermentation --n-components 6 \
    --method vip --top 15 --plot bands.png
```
VIP has a natural `>1` cut-off; SR gives sharper peaks without a universal
threshold. Use these to interpret the model or to prune bands — then re-fit and
**re-validate** the reduced model (band selection is part of the model and must
not leak into the test set).

## Order & leakage discipline

Split first → build one `Pipeline` (preprocess + model) → select LVs by CV on
train → fit on train → hand to **chemometric-validation** for held-out metrics
and to **model-diagnostics** for the applicability-domain check before trusting
predictions.

## Scripts

| script | purpose |
|---|---|
| `scripts/fit_model.py` | Build & fit a preprocess+model `Pipeline` (regression / classification / pca); persist it. Defines the `PLSDA` estimator. |
| `scripts/select_components.py` | CV sweep over `n_components`; min + 1-SE optima; RMSECV/accuracy-vs-LV plot. |
| `scripts/feature_importance.py` | VIP / SR band importance from a fitted PLS; top-band report + plot. |

Reference depth in `references/`; the decision tree in
`assets/modeling_decision.svg`; a copy-paste starter in
`assets/end_to_end_template.py`.
