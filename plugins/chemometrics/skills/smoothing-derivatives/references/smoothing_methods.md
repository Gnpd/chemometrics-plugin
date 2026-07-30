# Smoothing / denoising methods

Smoothing suppresses high-frequency measurement noise. The universal tradeoff:
more smoothing = less noise **and** less resolution (peaks broaden and flatten).
The goal is the *least* smoothing that gets noise low enough — never smoother
"to be safe."

All are in `chemotools.smooth`.

## Savitzky-Golay — `SavitzkyGolayFilter` (default choice)

Fits a low-order polynomial in a sliding window and evaluates it at the center.
Preserves peak height/position far better than a moving average.

- `window_length` (odd, `> polyorder`): the window width. **The main knob.**
  Keep it narrower than the narrowest real peak, or you flatten peaks.
- `polyorder`: polynomial degree (2 is typical). Higher order follows peaks more
  closely (less smoothing) for a given window.
- `mode`: boundary handling (`'nearest'`, `'mirror'`, `'interp'`, ...).

Tune with `scripts/tune_smoothing.py`.

## Whittaker — `WhittakerSmooth`

Penalized-least-squares smoother with a single continuous smoothness parameter
`lam` (larger = smoother). Global rather than windowed; no window-parity
constraint. Good when you want one smoothness dial. `solver_type='banded'`
(default) is fast; `'sparse'` is a numerical fallback.

## Moving average — `MeanFilter`

Simple box average over `window_length`. Cheap but blurs peaks the most; prefer
Savitzky-Golay unless you specifically want a box filter.

## Median filter — `MedianFilter`

Replaces each point with the window median. **The despiking tool**: removes
cosmic-ray spikes (Raman) and single-point outliers without the smearing a mean
filter causes. Often run *first*, before baseline correction.

## Modified sinc — `ModifiedSincFilter`

Modified sinc kernel (Schmid et al.): a flat passband and sharp frequency cutoff
(`window_length`, `n`, `alpha`). Use when you want frequency-selective smoothing
that removes noise above a cutoff while leaving broader features untouched.

## Choosing

```
General-purpose, preserve peaks        -> SavitzkyGolayFilter
One continuous smoothness knob          -> WhittakerSmooth
Cosmic-ray / spike removal (Raman)      -> MedianFilter (first)
Frequency-selective, sharp cutoff       -> ModifiedSincFilter
(Avoid MeanFilter unless you want a box average.)
```

## Order

Smoothing goes after baseline/scatter and before/with the derivative. If you
plan a derivative, prefer the Savitzky-Golay **derivative** (which smooths and
differentiates in one pass with one set of parameters) over a separate smoothing
step — see `derivative_methods.md`.
