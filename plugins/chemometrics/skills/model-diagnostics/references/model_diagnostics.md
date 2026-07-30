# Reading the inspector diagnostics

`diagnose_model.py` builds a chemotools `PCAInspector` / `PLSRegressionInspector`
and saves the figures below. The inspector is **experimental** (emits a
`FutureWarning`, API may change); the script suppresses the warning and always
prints a text `summary()` so the core numbers survive regardless.

## PLS regression figures

### Predicted vs. actual
Points on the 1:1 line = perfect prediction. Read for:
- **Scatter** about the line → random error (RMSEP).
- **Slope ≠ 1** → the model systematically compresses/expands the range (often
  too few LVs, "regression to the mean").
- **Offset from the line** → bias (see `chemometric-validation`).
- **Curvature** → a non-linear relationship the linear PLS cannot capture
  (consider a transform or more LVs).

### Residuals plot
Residual (ŷ − y) vs. predicted or vs. sample index. Should be a structureless
band around zero. Watch for:
- **Funnel shape** (residuals grow with the value) → heteroscedastic error;
  consider weighting or a transform.
- **Trend vs. index/time** → drift; a calibration-transfer or batch effect.
- **Isolated large residuals** → y-outliers; cross-check with studentized
  residuals and the reference values.

### Residual distribution / QQ
Residuals should be ~normal and centered at zero. Heavy tails or skew indicate
outliers or a missing systematic term.

### Latent scores (with T²/Q limits)
Score scatter (LV1 vs LV2, …) with the Hotelling T² ellipse and Q limits drawn.
Samples outside the ellipse/limits are the applicability-domain flags — see
`applicability_domain.md`. Clusters hint at unmodelled groups (batches,
instruments) that may warrant grouped CV or separate models.

## PCA figures

### Scores
The map of sample-to-sample similarity. Clusters = groups; lone points far from
the mass = outliers (confirm with T²/Q). Colour by a known factor (batch, class)
to see whether it dominates the variance.

### Loadings
Which wavelengths drive each component. Peaks align loadings with real spectral
bands — a sanity check that the model uses chemistry, not artefacts. A loading
dominated by a baseline slope or a single noisy channel is a preprocessing smell.

### Explained variance
Variance captured per component and cumulatively. The "elbow" suggests how many
components carry real structure vs. noise — a cross-check on the LV count chosen
by CV in `chemometric-modeling`.

## Numbers in the text summary

`text_summary()` reports `n_components`, sample/feature counts, train R²/RMSE, and
the Hotelling T²/Q-residual limits. Use these when running headless (no plots) or
in logs. For the metrics you *report*, use `chemometric-validation` on a held-out
set — the inspector's train figures are for diagnosis, not for claiming
performance.
