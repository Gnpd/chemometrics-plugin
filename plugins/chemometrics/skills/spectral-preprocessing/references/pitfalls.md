# Pitfalls

The recurring ways spectral preprocessing goes wrong, and how to avoid each.

## 1. Data leakage (the most consequential)

**Symptom:** optimistic cross-validation / test scores that collapse in
production.

**Cause:** fitting a data-learning step on data that includes the test set —
e.g. `StandardScaler().fit_transform(X_all)` before splitting, or calling
`pipeline.fit_transform(X_test)`.

**Fix:** build one `Pipeline`, `fit` it on training data only, `transform`
everything else. Steps that learn parameters and therefore leak if misused:
`StandardScaler`/`ParetoScaler` (column means/std), `MSC`/`EMSC` (reference
spectrum), `OSC`/`EPO`/`OrthogonalPLS` (projections), `AirPLS`/`ArPLS`/`AsLs`
(warm-start weights). Row-wise steps (SNV, RNV, min-max, norm, derivatives)
don't leak across samples, but keeping everything in the pipeline is still the
safe default.

## 2. Wrong array shape

**Symptom:** `ValueError` about 2-D input, or a single spectrum treated as many
one-point samples.

**Cause:** passing a 1-D array `(n_features,)`.

**Fix:** reshape to `(1, n_features)` (`x.reshape(1, -1)`). `load_spectra.py`
does this automatically. chemotools/scikit-learn always expect
`(n_samples, n_features)`.

## 3. Losing / mismatching the x-axis

**Symptom:** `range_cut` keeps the wrong region; band-based scalers point at the
wrong wavenumber; peaks misaligned.

**Cause:** the wavenumber/wavelength array lived only in the CSV header and was
dropped, or `start`/`end` were given as indices when the transformer expected
axis units (or vice-versa).

**Fix:** recover the axis (`load_spectra.py` parses float column headers) and
pass it via `$x_axis` in the spec. If you have no axis, give **integer indices**
and say so. Remember `range_cut` changes `n_features` — later index-based steps
must account for the new length.

## 4. NaNs from interpolation fill values

**Symptom:** NaNs appear after `XAxisInterpolator`, then every downstream model
fails.

**Cause:** `XAxisInterpolator` fills points outside the input grid with `left` /
`right`, which **default to NaN**.

**Fix:** set `left=0`/`right=0` (or another sentinel your model tolerates), or
choose a `common_x_axis` fully inside every input grid. `diagnose.py` flags
non-finite values per step. See the calibration-transfer skill.

## 5. Over-smoothing / over-differentiating

**Symptom:** peaks flattened or shifted; derivative output is mostly noise.

**Cause:** smoothing `window_length` too large relative to peak width; derivative
order too high; smoothing *and* a 2nd derivative *and* scatter correction all
stacked.

**Fix:** keep the smoothing window below the narrowest peak's width; prefer a
single Savitzky-Golay derivative (smoothing built in) over separate smooth +
derivative; use the minimal set of corrections. Tune with the
smoothing-derivatives skill's `tune_smoothing.py`.

## 6. Double normalization / redundant corrections

**Symptom:** signal squashed to noise; features near-constant.

**Cause:** e.g. SNV then MSC then autoscale; or baseline correction that already
normalized, followed by another scatter step.

**Fix:** one scatter/normalization step is almost always enough. If mean-centering
after SNV, use `StandardScaler(with_std=False)` — don't autoscale
already-normalized rows unless you mean to.

## 7. Applying corrections in the wrong domain

**Symptom:** nonlinear model behavior; poor quantitative fit.

**Cause:** correcting in transmittance when the chemistry is linear in
absorbance; forgetting Kubelka-Munk for diffuse reflectance.

**Fix:** convert units **first** (`IntensityConversion`). See `ordering_guide.md`.

## 8. Negative values where the model assumes non-negativity

**Symptom:** downstream method (e.g. some NMF/embedding) errors or misbehaves.

**Cause:** baseline subtraction / derivatives legitimately produce negatives.

**Fix:** only clip with `NonNegative` when the downstream step truly requires it
— don't clip derivative output by reflex; negatives there are real signal.

## 9. Order mistakes

**Symptom:** scatter correction distorted by residual baseline; scaler fit on
un-corrected data; region cut after a global normalization.

**Fix:** follow `ordering_guide.md`. Run `diagnose.py` to see each step's effect
cumulatively — a step that zeroes the std or blows up the range is a red flag.
