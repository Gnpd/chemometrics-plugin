---
name: baseline-scatter-correction
description: >-
  Remove baselines and scatter/multiplicative effects from spectra with chemotools.
  Use when spectra show sloping or curved baselines, fluorescence backgrounds (Raman),
  particle-size or path-length scatter, or additive/multiplicative offsets between
  spectra — and you must choose among AirPLS/ArPLS/AsLS/polynomial/rubberband baseline
  methods and SNV/RNV/MSC/EMSC scatter corrections, and tune their parameters (especially
  the baseline smoothness lam).
---

# Baseline & scatter correction

The two corrections that most affect model quality, and the two most often
misapplied. This skill helps choose the method and tune it. It is a spoke of the
**spectral-preprocessing** hub — return there for the overall pipeline and order.

## First: baseline or scatter (or both)?

Plot or inspect the raw spectra:

- **Additive, slowly-varying background** under the peaks — a slope, a curve, a
  broad hump (Raman fluorescence). Spectra don't sit on a common zero.
  → **baseline correction**.
- **Spectra roughly parallel but vertically offset and/or scaled** — same shape,
  different level, from particle size / path length / probe contact.
  → **scatter correction**.
- Often you need **one** of these, sometimes both (baseline first). A 1st/2nd
  derivative removes additive/linear baselines, so a derivative-based pipeline
  may not need a separate baseline step. Avoid stacking baseline + scatter + 2nd
  derivative by reflex — that over-processes.

`assets/baseline_scatter_decision.svg` summarizes the choice.

## Baseline: choosing and tuning

Read `references/baseline_methods.md`. Short version:

- **Automatic, smooth/curved drift or Raman fluorescence** → `AirPls` (default)
  or `ArPls`. The one knob that matters is `lam`.
- **Broad convex background, no baseline points** → `RubberbandCorrection`.
- **You can mark signal-free baseline regions** → `PolynomialCorrection` /
  `CubicSplineCorrection` with `indices`.
- **Simple linear tilt** → `LinearCorrection`.

**Tune `lam`** with `scripts/tune_baseline.py`:
```
python scripts/tune_baseline.py --input spectra.csv --method airpls
```
Pick the smallest `lam` (start `1e5`, step by ×10) that keeps `neg_fraction` low
(not carving into peaks) while flattening the background. Then confirm visually.

**Raman:** run a `MedianFilter` first to remove cosmic-ray spikes — otherwise
they corrupt the baseline fit.

## Scatter: choosing

Read `references/scatter_methods.md`. Short version:

- **Default, robust, no reference** → `StandardNormalVariate` (SNV).
- **SNV distorted by dominant peaks/outliers** → `RobustNormalVariate` (RNV).
- **A meaningful reference spectrum exists** → `MultiplicativeScatterCorrection`.
- **Wavelength-dependent scatter or known interferents** →
  `ExtendedMultiplicativeScatterCorrection`.

Use **one** scatter method. `MSC`/`EMSC` learn a reference from training data —
keep them inside the Pipeline and fit on train only (leakage otherwise).

Compare options on your data:
```
python scripts/compare_corrections.py --input spectra.csv --out compare.png
```

## Order (reminder)

`baseline → scatter`, both before smoothing/derivative/scaling. Full ordering
rationale is in the hub's `references/ordering_guide.md`.

## Scripts

| script | purpose |
|---|---|
| `scripts/tune_baseline.py` | Sweep `lam` for AirPLS/ArPLS/AsLs; report roughness / over-subtraction tradeoff. |
| `scripts/compare_corrections.py` | Apply a menu of baseline/scatter options; compare metrics + optional figure. |
