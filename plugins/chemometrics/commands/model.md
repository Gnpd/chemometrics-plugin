---
description: Fit a chemometric model (PCA / PLS regression / PLS-DA) and choose the number of latent variables by CV.
argument-hint: "[dataset name or path] [regression|classification|pca]"
---

Build and fit a chemometric model by following the **chemometric-modeling**
skill. Load that skill and apply its guidance; delegate preprocessing to
`spectral-preprocessing` and evaluation to `chemometric-validation`.

Target: **$ARGUMENTS**
(A bundled dataset — `fermentation` (regression) or `coffee` (classification) —
or a path to spectra, optionally followed by the task.)

Do:

1. Confirm the task and whether a preprocessing head is wanted; **split first**.
2. Choose the number of latent variables by cross-validation (the **1-SE rule**),
   not by training fit — `skills/chemometric-modeling/scripts/select_components.py`.
3. Fit the preprocess+model `Pipeline` on train —
   `skills/chemometric-modeling/scripts/fit_model.py` — and report train fit plus
   the CV-selected LV count.
4. Optionally rank informative bands with VIP/SR
   (`feature_importance.py`); if you prune bands, re-fit and re-validate.
5. Recommend the next step: validate with `/chemometrics:validate`, and gate
   predictions with the applicability domain (`model-diagnostics`).

Report the pipeline, the chosen LVs and why, and any band-selection findings.
Do not claim performance from the training fit — that is what validation is for.
