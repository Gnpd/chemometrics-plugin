# Scatter correction / normalization methods

Scatter effects are **multiplicative** (and additive) per-spectrum distortions
from physical rather than chemical causes: particle size, sample packing,
path-length, probe contact. They make otherwise-identical spectra sit at
different offsets and scales. Scatter correction removes that nuisance variation
so the model sees chemistry, not physics.

All are in `chemotools.scatter`. SNV/RNV are row-wise and need no reference;
MSC/EMSC regress against a reference spectrum learned at `fit`.

## Standard Normal Variate — `StandardNormalVariate` (`snv`)

Per spectrum: subtract its mean, divide by its standard deviation.

- No reference, no parameters, no fitting across samples → cannot leak.
- Removes both additive offset and multiplicative scale per spectrum.
- The default first choice. Very widely used in NIR/IR.

## Robust Normal Variate — `RobustNormalVariate` (`rnv`, `percentile=25`)

Like SNV but centers/scales using a percentile instead of mean/std, so strong
peaks or outlier channels don't dominate the normalization. Use when SNV is
distorted by a few large bands.

## Multiplicative Scatter Correction — `MultiplicativeScatterCorrection` (`msc`)

Regress each spectrum onto a **reference** (the mean spectrum by default, or
`method='median'`, or a supplied `reference`) and remove the fitted slope
(multiplicative) and intercept (additive).

- Needs a representative reference. `fit` learns the mean spectrum from the
  **training** set; that same reference is then applied to test spectra — so MSC
  must live inside the Pipeline and be fit on train only (leakage risk otherwise).
- Comparable to SNV in effect; MSC ties every spectrum to a common reference,
  which can be preferable when a physically meaningful reference exists.

## Extended MSC — `ExtendedMultiplicativeScatterCorrection` (`emsc`, `order=2`)

MSC plus polynomial terms (degree `order`) and optional `interferences` spectra.
Removes **wavelength-dependent** scatter and known interferents (e.g. a solvent
or water signature) that plain MSC/SNV cannot. Use when scatter varies across
the spectrum or you have spectra of known interferences to project out.

## SNV vs MSC vs EMSC vs RNV

```
No reference, simplest, robust default      -> SNV
SNV distorted by dominant peaks/outliers    -> RNV
Meaningful reference spectrum exists         -> MSC
Wavelength-dependent scatter / known
  interferents to remove                     -> EMSC
```

In practice SNV and MSC give similar model performance; try both. Use **one**
scatter correction — stacking SNV then MSC is redundant (see hub `pitfalls.md`).

## Order & leakage

- Apply **after** baseline correction (or rely on a derivative to remove the
  baseline). Residual baseline leaks into the mean/scale these estimate.
- MSC/EMSC learn a reference from data → keep them in the Pipeline, fit on train.
  SNV/RNV are row-wise and safe either way, but keeping everything in one
  Pipeline is the right default.
- `scripts/compare_corrections.py` shows the effect of each on your spectra.
