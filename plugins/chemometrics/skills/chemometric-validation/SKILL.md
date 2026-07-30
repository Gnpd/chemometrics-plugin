---
name: chemometric-validation
description: >-
  Validate chemometric models rigorously with scikit-learn. Use when cross-validating a
  spectra→property model without leakage (grouped folds for replicate or augmented spectra),
  reporting chemometric metrics (RMSECV, RMSEP, R², RPD, bias) or classification metrics
  (accuracy, F1, confusion matrix), running a permutation / y-scramble significance test, or
  diagnosing over- and under-fitting with component and learning curves.
---

# Chemometric validation

Prove a model generalizes — and prove its score is not chance. This skill owns
the *validate* stage: leakage-free cross-validation, chemometric metrics, and a
significance test. Model building lives in **chemometric-modeling**;
applicability-domain / prediction-time QC lives in **model-diagnostics**.

## The one rule: validate the whole pipeline, on train only

Cross-validate the **entire** `Pipeline` (preprocessing + model), so every fold
refits preprocessing on its own training rows. Fitting preprocessing (SNV means,
scaler statistics, PLS loadings) on all data *before* CV leaks the test set and
inflates every metric. `cross_validate.py` and `permutation_test.py` both refit
per fold. Details and the common leakage traps: `references/cv_strategies.md`.

## Choose the fold strategy (`assets/validation_decision.svg`)

- **Grouped** — replicate or augmented spectra of one physical sample? Use
  `GroupKFold` (`--groups`) so copies never straddle a fold. Skipping this is the
  most common way to get a beautiful-but-fake CV score; the augmentation skill's
  copies *require* it.
- **Stratified** — classification: `StratifiedKFold` keeps class balance per fold.
- **Plain KFold** — regression without groups.

```
# grouped CV of a preprocess+PLS pipeline
python scripts/cross_validate.py --task regression --dataset fermentation \
    --n-components 6 --cv 5 --groups groups.npy
```
Reports each metric as `mean ± std` across folds — the spread is part of the
result. `run_cv` is importable.

## Metrics that matter

Report the chemometric set, not a single number (`references/metrics_reference.md`,
`assets/metrics_cheatsheet.md`):

- **Regression:** RMSEP (units of y), R², **RPD** (`std(y)/RMSEP`; >2 useful, >3
  good), **bias** (systematic offset — a transfer/drift red flag), SEP.
- **Classification:** accuracy, macro-F1, confusion matrix.

```
python scripts/metrics.py --task regression --dataset fermentation --model pls.joblib
```
`regression_metrics` / `classification_metrics` are importable helpers. Only the
chemometric aggregates are wrapped here; everything else defers to `sklearn.metrics`.

## Is it real? — permutation test

Spectra are wide (bands ≫ samples), so a model can fit random labels. Confirm the
signal with a **y-scramble** test: refit on shuffled targets to build the null,
then compare the true CV score.

```
python scripts/permutation_test.py --task classification --dataset coffee \
    --n-components 4 --n-permutations 200 --plot perm.png
```
A small p-value (< 0.05) means the model beats chance. `permutation_test` is
importable.

## Over- vs under-fitting

The component curve (`chemometric-modeling/select_components.py`) is the primary
tool: CV error that keeps rising past the minimum while training error falls = too
many LVs. A large train-vs-CV gap = over-fit; both high and close = under-fit.
`references/cv_strategies.md` covers nested CV when you also *tune* hyperparameters
(tune inside an inner loop; report on the outer).

## Order & leakage discipline

Split first → CV the whole pipeline on train (grouped where needed) → confirm
significance with the permutation test → report held-out RMSEP/metrics on the
untouched test set → then **model-diagnostics** guards each prediction with the
applicability domain.

## Scripts

| script | purpose |
|---|---|
| `scripts/cross_validate.py` | Leakage-free `cross_validate` of a full pipeline; grouped/stratified/plain folds; `mean ± std` per metric. |
| `scripts/permutation_test.py` | y-scramble null distribution + p-value; histogram plot. |
| `scripts/metrics.py` | RMSEP/R²/RPD/bias/SEP or accuracy/F1/confusion on a held-out set; importable. |
