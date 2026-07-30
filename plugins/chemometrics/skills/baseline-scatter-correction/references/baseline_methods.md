# Baseline correction methods

A baseline is an additive, slowly-varying background under the peaks:
instrumental drift, ATR contact, sloping detector response, or (in Raman) a
broad fluorescence hump. Baseline correction estimates that background and
subtracts it. Choose by **how much you know about the baseline's shape** and
**whether you can identify baseline regions**.

## Automatic penalized-least-squares family (no baseline points needed)

These fit a smooth curve to the non-peak part of the signal by iteratively
down-weighting points that look like peaks. Controlled by `lam` (smoothness):
larger `lam` → stiffer, smoother baseline.

| class | extra knob | notes |
|---|---|---|
| `AirPls` | `nr_iterations` | Adaptive reweighting. Best default for smooth/curved backgrounds and Raman fluorescence. Robust, little tuning beyond `lam`. |
| `ArPls` | `ratio` | Asymmetrically reweighted. Often more stable than AsLs when peaks sit on noisy baselines. |
| `AsLs` | `penalty` (asymmetry `p`) | The original Eilers asymmetric least squares. Two knobs (`lam`, `penalty`) → more tuning, more control. |

**Tuning `lam`:** use `scripts/tune_baseline.py`. Start at `1e5`; move by
factors of 10. Pick the *smallest* `lam` that keeps the corrected signal from
going substantially negative (over-subtraction) while still flattening the
background. Too-large `lam` leaves residual drift; too-small `lam` carves into
peaks.

All three support a fast `banded` solver (default) and a `sparse` fallback for
ill-conditioned problems, and `n_jobs` for row-parallel transform.

## Deterministic methods (you identify the baseline)

| class | when |
|---|---|
| `LinearCorrection` | Baseline is a straight tilt between the spectrum endpoints. Fast, assumption-heavy. |
| `PolynomialCorrection` (`order`, `indices`) | Fit a polynomial through channels you mark as baseline (`indices`). Good when clear baseline regions exist between bands. |
| `CubicSplineCorrection` (`indices`) | Spline through baseline points; follows curved backgrounds more flexibly than a polynomial. |
| `ConstantBaselineCorrection` (`start`, `end`, `x_axis`) | Subtract the mean of a flat, signal-free region. |
| `RubberbandCorrection` | Convex-hull "rubber band" under the spectrum. Parameter-free; good for broadly convex backgrounds, common in Raman/IR. |
| `SubtractReference` | Subtract a measured background/reference spectrum. For a *per-sample* background applied at transform time, see the calibration-transfer skill. |

## Cleanup

`NonNegative` (`mode='zero'|'abs'`) clips residual negatives after correction —
use only when a downstream step requires non-negativity; do **not** apply it to
derivative output (negatives there are real).

## Choosing

```
Broad convex background, no obvious baseline points   -> RubberbandCorrection or AirPLS
Smooth/curved drift, automatic                        -> AirPLS (or ArPLS)
Raman fluorescence hump                               -> AirPLS / ArPLS (median-filter spikes first)
Clear signal-free baseline regions                    -> PolynomialCorrection / CubicSplineCorrection (indices)
Simple linear tilt                                    -> LinearCorrection
Measured background available                         -> SubtractReference
```

## Order note

Baseline correction comes **before** scatter correction (SNV/MSC assume the
additive background is gone). A 1st/2nd derivative also removes additive/linear
baselines — if you derivative, you often don't need a separate baseline step.
See the hub `ordering_guide.md`.
