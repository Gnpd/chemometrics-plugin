---
name: model-diagnostics
description: >-
  Diagnose fitted PCA/PLS chemometric models and define an applicability domain with chemotools.
  Use to decide whether a new spectrum is safe to predict (Hotelling T², Q-residuals, leverage,
  DModX, studentized residuals), to flag outliers and extrapolation at prediction time, and to
  inspect a model visually (scores, loadings, predicted-vs-actual, residuals) via the chemotools
  inspector and plotting modules.
---

# Model diagnostics & applicability domain

Two prediction-time questions this skill answers: **"is this spectrum safe to
predict?"** (applicability domain) and **"what is my fitted model doing?"**
(visual diagnostics). These use chemotools' unique `outliers` and `inspector`
modules — there is no sklearn equivalent. Model building is
**chemometric-modeling**; generalization testing is **chemometric-validation**.

## Applicability domain — the prediction-time gate

A model only extrapolates safely inside the spectral region it was trained on. A
prediction for a sample *outside* that region is unsupported no matter how
confident the number looks — a new interferent, a fouled probe, a different
matrix. The applicability domain (AD) makes that explicit.

Fit the AD on the **training** model, then flag new spectra:
```
python scripts/applicability_domain.py --dataset fermentation --n-components 5 \
    --confidence 0.95 --test new_spectra.csv
```

Two complementary distances (`assets/applicability_domain_decision.svg`,
`references/applicability_domain.md`):

- **Hotelling T²** — distance *within* the model plane. Large = unusual but
  in-model scores (an extreme concentration).
- **Q-residuals** — distance *off* the model plane. Large = spectral features the
  model cannot represent (a genuinely new kind of sample). This is the one that
  catches "never seen anything like this".

A sample outside **either** limit should not be trusted. `fit_domain` / `flag`
are importable so the gate drops into a prediction service (score → AD check →
return value *or* "out of domain, recalibrate"). Leverage, DModX, and studentized
residuals are covered in the reference for when you need them.

## Visual diagnostics — inspect the model

Look at what the model learned:
```
python scripts/diagnose_model.py --task regression --dataset fermentation \
    --n-components 5 --outdir figs/
```
Emits predicted-vs-actual, residuals, residual distribution, and latent-score
figures (PLS); scores/loadings/variance (PCA). Always prints a text `summary()`
so core numbers (R², RMSE, T²/Q limits) survive even if plotting is unavailable.
Reading the plots: `references/model_diagnostics.md`.

**The inspector is experimental** — it emits a `FutureWarning` and its API may
change. The script suppresses the warning, degrades gracefully if the inspector
or matplotlib is missing, and never lets a plotting failure lose the numbers. Use
`applicability_domain.py` (not the inspector) for the automated go/no-go gate;
the inspector is for human inspection.

## Order & leakage discipline

The AD is fit on **train only**, like everything else — it is part of the
deployed model. Sequence: model (chemometric-modeling) → validate
(chemometric-validation) → **fit AD on train** → at prediction time, AD-gate every
new spectrum before returning a value. Never fit the AD on the samples you are
about to judge.

## Scripts

| script | purpose |
|---|---|
| `scripts/applicability_domain.py` | Fit Hotelling T² + Q-residuals on the training model; flag out-of-domain samples with counts + indices; importable gate. |
| `scripts/diagnose_model.py` | Build the chemotools inspector; save scores/loadings/predicted-vs-actual/residual figures; text summary fallback; suppresses the experimental `FutureWarning`. |
