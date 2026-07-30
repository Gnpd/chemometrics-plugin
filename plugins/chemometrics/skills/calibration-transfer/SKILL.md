---
name: calibration-transfer
description: >-
  Align and transfer spectra between instruments or measurement conditions with chemotools
  adaptation methods. Use when spectra sit on different wavenumber/wavelength grids, when a
  model trained on one instrument must run on another, or when per-measurement metadata
  (a fresh background, an x-axis, laser power) must be applied at transform time via
  scikit-learn metadata routing. Covers XAxisInterpolator, DirectStandardization, and
  PiecewiseDirectStandardization.
---

# Calibration transfer & axis alignment

Make spectra from different instruments, sessions, or grids comparable — and
apply per-measurement information at transform time. A spoke of the
**spectral-preprocessing** hub; these steps come **early**, before the standard
correction chain.

## Which problem do you have?

Diagnose first (`assets/calibration_transfer_decision.svg`):

1. **Different x-axis grids** — the wavenumber/wavelength arrays differ, so
   spectra can't even be stacked into one matrix. Peaks that should align land
   in different columns. → **align** with `XAxisInterpolator`.
2. **Instrument response difference** — same grid, but a model trained on
   instrument A degrades on instrument B (different gain/resolution/baseline).
   → **standardize** with `DirectStandardization` (uniform difference) or
   `PiecewiseDirectStandardization` (wavelength-dependent).
3. **Per-measurement metadata** at transform time (fresh background, per-sample
   axis, scale factor) → **metadata routing** (below).

Read `references/adaptation_methods.md` for the method details.

## Aligning grids — XAxisInterpolator

Resample every spectrum onto a `common_x_axis`:
```
python scripts/align_axes.py --input spectra.csv --num 1000 --method linear --fill 0 --out aligned.csv
```
Key choices: `method` (`linear`/`cubic`/`pchip`) and the **fill value** for points
outside an input grid — `left`/`right` **default to NaN**, so set `--fill 0` (or
keep the common grid inside every input grid) or every downstream step gets NaNs.
This is the most common failure here (hub `pitfalls.md` §4).

## Standardizing instruments — DS / PDS

You need **transfer standards**: samples measured on both instruments. Fit maps
target-instrument spectra onto the source instrument's space:
```python
ds = DirectStandardization().fit(X_target_std, X_source=X_source_std)
X_mapped = ds.transform(X_target_new)
```
- Uniform difference across the spectrum → `DirectStandardization`.
- Wavelength-dependent difference → `PiecewiseDirectStandardization`
  (`window_length`).

Compare both on synthetic instruments:
```
python scripts/transfer_demo.py
```

## Metadata routing

For per-call information, enable routing and declare the requests:
```python
import sklearn
sklearn.set_config(enable_metadata_routing=True)
interp = (XAxisInterpolator(common_x_axis=x_common, method="linear", left=0, right=0)
          .set_fit_request(x_axis=True).set_transform_request(x_axis=True))
X_aligned = interp.fit_transform(X, x_axis=per_sample_axes)
```
In a Pipeline the metadata reaches **only** the step that declared it. Full
treatment (shared vs. per-sample shapes, pipeline example) in
`references/metadata_routing.md`.

## Order

Alignment → standardization → (then the usual baseline/scatter/smooth/derivative/
scale chain from the hub). Everything downstream assumes a common grid and a
common instrument response.

## Scripts

| script | purpose |
|---|---|
| `scripts/align_axes.py` | Resample spectra onto a common grid via `XAxisInterpolator` + metadata routing. |
| `scripts/transfer_demo.py` | Self-contained DS vs. PDS instrument-standardization demo with RMSE before/after. |
