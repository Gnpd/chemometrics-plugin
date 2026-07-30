# Derivative methods

Spectral derivatives do two useful things at once: **remove baselines**
(1st derivative kills a constant offset, 2nd derivative kills a linear slope)
and **enhance resolution** (sharpen and separate overlapping bands). The cost is
**noise amplification** — differentiation multiplies high-frequency noise — so
derivatives are always paired with smoothing.

All are in `chemotools.derivative`.

## Savitzky-Golay derivative — `SavitzkyGolay` (default choice)

Computes the derivative of the fitted local polynomial — smoothing and
differentiation in a **single pass**, one parameter set. Prefer this over
smoothing then differentiating separately.

- `deriv`: derivative order (0 = smoothing only, 1, 2, ...).
- `window_length` (odd): larger windows tame the noise the derivative amplifies.
  A 2nd derivative needs a larger window than a 1st for the same noise level.
- `polyorder`: must be **greater than `deriv`** (e.g. `polyorder>=2` for
  `deriv=2`). 2 is a common default; 3 preserves sharp peaks better.

## Norris-Williams — `NorrisWilliams`

Gap-segment derivative: smooths over `window_length` then differences across a
`gap_size`. The classic NIR derivative. `deriv` selects order. Use when matching
established NIR workflows or when the gap-segment behavior is desired.

## 1st vs 2nd derivative

| | removes | resolves | noise | typical use |
|---|---|---|---|---|
| 1st (`deriv=1`) | constant offset | some overlap, sharpens | amplified | general baseline+sharpening |
| 2nd (`deriv=2`) | constant + linear baseline | strongly separates overlapping bands (peaks → negative lobes) | strongly amplified | NIR overtone resolution |

2nd-derivative peaks point *downward* (a band becomes a negative trough between
two positive lobes) — expected, not a bug. Don't `NonNegative`-clip derivative
output; the negatives are real signal.

## Tuning

Use `scripts/derivative_preview.py` to view `deriv` 0/1/2 with a chosen
window/polyorder and compare noise. Increase `window_length` until the
derivative is smooth enough without flattening the features you care about
(`scripts/tune_smoothing.py` tunes the same window for the underlying smoothing).

## Order & interaction with baseline

Because a derivative removes additive/linear baselines, a derivative-based
pipeline **often needs no separate baseline correction step**. A common,
compact recipe is `scatter (SNV) → SavitzkyGolay(deriv=1 or 2) → mean-center`,
skipping explicit baseline correction. See the hub `ordering_guide.md`.
