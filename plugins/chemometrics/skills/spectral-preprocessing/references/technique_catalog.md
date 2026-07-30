# Technique catalog

Every chemotools preprocessing transformer, grouped by purpose. Import paths,
key parameters (with current, non-deprecated names and defaults), and when to
reach for each. All are scikit-learn transformers: `fit(X)` then `transform(X)`,
`X` of shape `(n_samples, n_features)`. `build_pipeline.py` exposes each under
the short `type` name in the first column.

> Parameters shown are the *decision-relevant* ones. Every transformer also
> validates its arguments via scikit-learn `_parameter_constraints`; consult the
> class docstring (`help(Class)`) for the full list and exact ranges.

## Physics / unit conversion — `chemotools.physics`

| type | class | key params | use when |
|---|---|---|---|
| `intensity_conversion` | `IntensityConversion` | `input_unit`, `output_unit` ∈ {`absorbance`, `transmittance`, `reflectance`, `kubelka_munk`, `pseudoabsorbance`} | Convert intensity representation. Usually the **first** step, so every later correction operates in the right domain (e.g. reflectance → Kubelka-Munk for diffuse NIR, transmittance → absorbance for Beer-Lambert linearity). |

## Region selection — `chemotools.feature_selection`

| type | class | key params | use when |
|---|---|---|---|
| `range_cut` | `RangeCut` | `start`, `end`, `x_axis` | Keep a contiguous spectral window. If `x_axis` is given, `start`/`end` are in axis units (e.g. cm⁻¹); otherwise they are integer indices. Do this **early** to drop uninformative/noisy regions (CO₂, water, detector roll-off). Changes `n_features`. |
| `index_selector` | `IndexSelector` | `features`, `x_axis` | Keep an arbitrary (non-contiguous) set of bands. |
| — | `SRSelector` | selectivity-ratio based | Supervised band selection from a fitted PLS model (advanced). |
| — | `VIPSelector` | VIP-score based | Supervised band selection (Variable Importance in Projection). |

## Baseline correction — `chemotools.baseline`

| type | class | key params | use when |
|---|---|---|---|
| `airpls` | `AirPls` | `lam=1e4`, `nr_iterations=100`, `solver_type='banded'`, `n_jobs=1` | Adaptive iteratively-reweighted penalized least squares. Best general-purpose automatic baseline for smoothly-varying backgrounds and Raman fluorescence. Larger `lam` → smoother/stiffer baseline. |
| `arpls` | `ArPls` | `lam=1e4`, `ratio=0.01` | Asymmetrically reweighted PLS. Similar to AirPLS, often more robust when peaks sit on noisy backgrounds. |
| `asls` | `AsLs` | `lam=1e4`, `penalty=0.01` | Asymmetric least squares (Eilers). The classic; `penalty` (`p`) sets asymmetry. Needs manual tuning of `lam`/`penalty`. |
| `polynomial_correction` | `PolynomialCorrection` | `order=1`, `indices=None` | Fit and subtract a polynomial through selected `indices` (baseline points). Deterministic; good when you can identify baseline regions. |
| `linear_correction` | `LinearCorrection` | — | Subtract a straight line through the endpoints. Fast, assumes linear tilt. |
| `constant_baseline_correction` | `ConstantBaselineCorrection` | `start`, `end`, `x_axis` | Subtract the mean of a flat reference region. |
| `cubic_spline_correction` | `CubicSplineCorrection` | `indices=None` | Spline through baseline points; flexible curved baselines. |
| `rubberband_correction` | `RubberbandCorrection` | `n_jobs=1` | Convex-hull ("rubber band") baseline. Parameter-free; good for broad convex backgrounds. |
| `non_negative` | `NonNegative` | `mode='zero'` \| `'abs'` | Clip negatives to 0 (or take abs). A cleanup step after correction, not a baseline estimator. |
| `subtract_reference` | `SubtractReference` | `reference`, `scale_reference`, `start`, `end`, `x_axis` | Subtract a measured reference/background spectrum. See also the dynamic `calibration-transfer` skill for subtracting a *per-sample* background at transform time. |

See the **baseline-scatter-correction** skill for choosing among these and tuning `lam`.

## Scatter correction / normalization — `chemotools.scatter`

| type | class | key params | use when |
|---|---|---|---|
| `snv` | `StandardNormalVariate` | — | Row-wise: subtract mean, divide by std. The default scatter correction; removes multiplicative + additive per-spectrum effects. No reference needed. |
| `rnv` | `RobustNormalVariate` | `percentile=25` | Robust SNV using a percentile instead of the mean; resists outlier bands/peaks. |
| `msc` | `MultiplicativeScatterCorrection` | `method='mean'`\|`'median'`, `reference=None` | Regress each spectrum onto a reference (mean spectrum by default) and correct slope/offset. Needs a representative reference; `fit` learns it from training data. |
| `emsc` | `ExtendedMultiplicativeScatterCorrection` | `order=2`, `reference=None`, `interferences=None` | MSC plus polynomial terms (and optional interferent spectra) to remove wavelength-dependent scatter and known interferences. |

See the **baseline-scatter-correction** skill for SNV vs MSC vs EMSC.

## Smoothing / denoising — `chemotools.smooth`

| type | class | key params | use when |
|---|---|---|---|
| `savgol_filter` | `SavitzkyGolayFilter` | `window_length=3`, `polyorder=1`, `mode='nearest'` | Polynomial (Savitzky-Golay) smoothing. The standard choice; preserves peak shape better than a moving average. `window_length` odd. |
| `whittaker` | `WhittakerSmooth` | `lam=1e4`, `solver_type='banded'` | Whittaker smoother; penalized least squares. `lam` controls smoothness continuously. |
| `mean_filter` | `MeanFilter` | `window_length=3` | Moving average. Simple; blurs peaks. |
| `median_filter` | `MedianFilter` | `window_length=3` | Median filter; removes spikes/cosmic rays (Raman) without smearing edges as much. |
| `modified_sinc_filter` | `ModifiedSincFilter` | `window_length=21`, `n=6`, `alpha=4.0` | Modified sinc (Schmid et al.); flat passband, sharp cutoff. Good frequency-selective smoothing. |

See the **smoothing-derivatives** skill for window/order selection.

## Derivatives — `chemotools.derivative`

| type | class | key params | use when |
|---|---|---|---|
| `savgol_deriv` | `SavitzkyGolay` | `window_length=3`, `polyorder=1`, `deriv=1`, `mode='nearest'` | Savitzky-Golay derivative (smoothing + differentiation in one pass). `deriv=1` removes additive baselines and sharpens; `deriv=2` also removes linear baselines and resolves overlapping bands (at the cost of SNR). |
| `norris_williams` | `NorrisWilliams` | `window_length=5`, `gap_size=3`, `deriv=1` | Norris-Williams gap-segment derivative. Classic NIR derivative with a smoothing gap. |

Derivatives **amplify noise** — smooth first or use the built-in smoothing of the SG derivative. See the **smoothing-derivatives** skill.

## Scaling — `chemotools.scale`

| type | class | key params | use when |
|---|---|---|---|
| `min_max_scaler` | `MinMaxScaler` | `use_min=True` | Row-wise min-max to [0, 1]. |
| `norm_scaler` | `NormScaler` | `l_norm=2` | Row-wise Lp normalization (L2 by default). |
| `pareto_scaler` | `ParetoScaler` | `p=0.5`, `with_mean=True` | Column-wise Pareto scaling (divide by √std); between mean-centering and autoscaling. Common in metabolomics. |
| `point_scaler` | `PointScaler` | `point=0`, `x_axis` | Normalize each spectrum to the intensity at one band/point. |
| `band_scaler` | `BandScaler` | `start`, `end`, `x_axis`, `aggregation='mean'` | Normalize to the aggregated intensity over a band. |
| `standard_scaler` | `sklearn.preprocessing.StandardScaler` | `with_mean`, `with_std` | Column-wise. **Mean-centering** (`with_mean=True, with_std=False`) is the usual last step before PCA/PLS. Full autoscaling sets `with_std=True`. Fitted on training data → applied to test. |

## Orthogonal signal correction — `chemotools.projection`

| type | class | use when |
|---|---|---|
| `osc` | `OrthogonalSignalCorrection` | Remove variance in `X` orthogonal (unrelated) to the target `y` before regression. Supervised; fit with `y`. |
| `direct_orthogonalization` | `DirectOrthogonalization` | Remove directions correlated with a known interference matrix. |
| `epo` | `ExternalParameterOrthogonalization` | Remove variance caused by an external parameter (temperature, humidity) using replicate spectra. |
| — | `OrthogonalPLS` | OPLS modeling. |

These are powerful but easy to misuse (they can remove real signal); apply after basic correction and validate carefully.

## Axis alignment / calibration transfer — `chemotools.adaptation`

| type | class | use when |
|---|---|---|
| `x_axis_interpolator` | `XAxisInterpolator` | Resample spectra onto a common x-axis grid (multi-instrument, drifted calibration). Uses metadata routing for per-sample axes. |
| — | `DirectStandardization` / `PiecewiseDirectStandardization` | Map spectra from a slave instrument onto a master instrument's response. |

See the dedicated **calibration-transfer** skill.

## Augmentation — `chemotools.augmentation`

`AddNoise`, `BaselineShift`, `FractionalShift`, `GaussianBroadening`, `IndexShift`,
`SpectrumScale` — synthetic variation for training robust models, **not** a
cleaning step. See the (phase-2) **spectral-augmentation** skill.
