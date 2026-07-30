# Ordering guide

Preprocessing order is not arbitrary — each step assumes the previous ones have
run. Applying them out of order silently degrades the result.

## The canonical order

```
1. Unit conversion      (physics.IntensityConversion)
2. Region selection     (feature_selection.RangeCut / IndexSelector)
3. Baseline correction  (baseline.*)          ─┐ pick the ones your artifact needs;
4. Scatter correction   (scatter.*)           ─┘ often only ONE of 3/4, sometimes both
5. Smoothing            (smooth.*)             ┐ frequently combined into a single
6. Derivative           (derivative.*)         ┘ Savitzky-Golay derivative step
7. Scaling / centering  (scale.* / StandardScaler)
8. (model: PCA / PLS / ...)
```

### Why this order

- **Unit conversion first.** Corrections assume a physically meaningful domain.
  Beer-Lambert linearity holds in absorbance, not transmittance; diffuse
  reflectance is linearized by Kubelka-Munk. Converting after correcting means
  correcting the wrong quantity.
- **Region cut early.** Removing dead/noisy regions before baseline and scatter
  means those steps aren't distorted by junk (e.g. a huge water band or detector
  roll-off dominating an SNV normalization). It also speeds everything up. Note
  it changes `n_features`, so anything referencing absolute indices must come
  after it (or use `x_axis` units).
- **Baseline before scatter.** Scatter corrections (SNV/MSC) assume the additive
  background is already gone; a residual baseline leaks into the mean/scale they
  estimate. If you use a derivative to remove the baseline instead, you may skip
  explicit baseline correction (see recipes).
- **Smooth before / within derivative.** Differentiation amplifies
  high-frequency noise. Either smooth first, or use `SavitzkyGolay` which folds
  smoothing into the derivative (preferred — one step, one set of parameters).
- **Scaling / centering last.** Mean-centering and (auto)scaling define the
  space the model sees; they must act on the fully corrected signal. Column-wise
  scalers (`StandardScaler`, `ParetoScaler`) **learn parameters from training
  data** and must be fit on train only (the Pipeline handles this).

### Where order is genuinely flexible

- Row-wise steps that don't learn cross-sample statistics (SNV, RNV, min-max,
  norm, point/band scaling) commute with each other in principle, but their
  *interaction with baseline/derivative* still fixes their place above.
- You rarely need both a baseline estimator **and** a full scatter correction
  **and** a 2nd derivative — that usually over-processes. Pick the minimal set.

## Fit on train, apply to test

Any step that learns from data — `MSC`/`EMSC` reference, `StandardScaler`
mean/std, `OSC`/`EPO` projections, `AirPLS` warm-start — must be `fit` on the
training set only and then used to `transform` validation/test/production
spectra. A `sklearn.pipeline.Pipeline` enforces this automatically: call
`pipeline.fit(X_train, y_train)` once, then `pipeline.transform(X_new)`. Never
call `fit_transform` on your test set. See `pitfalls.md`.

## Named recipes

Starting points, not laws — always verify against your data with `diagnose.py`
and `plot_spectra.py`.

### ATR-FTIR, quantitative (PLS regression) — the bundled fermentation case
```
range_cut (fingerprint region) -> airpls -> snv
  -> savgol_deriv(window_length=21, polyorder=2, deriv=1) -> standard_scaler(center)
```
This is `assets/pipeline_spec.yaml`. AirPLS removes the drifting ATR baseline;
SNV handles contact/path-length variation; 1st derivative sharpens; centering
prepares for PLS.

### NIR diffuse reflectance
```
intensity_conversion(reflectance -> kubelka_munk)   # or -> absorbance (log(1/R))
  -> snv (or msc) -> savgol_deriv(deriv=2) -> standard_scaler(center)
```
2nd derivative is common in NIR to resolve broad overlapping overtones and kill
linear baselines. SNV/MSC handle particle-size scatter.

### Raman with fluorescence background
```
median_filter (despike cosmic rays) -> airpls (or arpls) -> snv
  -> savgol_filter -> standard_scaler(center)
```
Median filter first removes cosmic-ray spikes that would otherwise corrupt the
baseline fit. AirPLS/ArPLS remove the broad fluorescence background.

### UV-Vis (relatively clean)
```
range_cut -> linear_correction (or polynomial_correction) -> standard_scaler
```
Often minimal preprocessing is enough; avoid over-processing.

### Classification (e.g. PLS-DA on the coffee dataset)
```
range_cut -> snv -> savgol_deriv(deriv=1) -> standard_scaler(center)
```
Emphasize between-class spectral shape; SNV + derivative remove nuisance
scatter/baseline that isn't class-discriminative.

## Deciding what your data needs

Look at the raw spectra (`plot_spectra.py` with an identity/near-empty spec, or
just plot the loaded array):

- **Sloping or curved offset that differs per spectrum** → baseline correction
  (AirPLS/ArPLS for smooth/curved; linear/polynomial if you know baseline points;
  rubberband for convex).
- **Spectra parallel but vertically offset / scaled** (multiplicative scatter,
  path length, particle size) → SNV or MSC.
- **Visible high-frequency noise** → smoothing (or a smoothing derivative).
- **Overlapping broad bands you need to resolve** → 2nd derivative.
- **Different instruments / shifted x-axis** → calibration-transfer skill.

The decision tree in `assets/decision_tree.svg` summarizes this.
