# Augmentation methods

Each transformer in `chemotools.augmentation` draws random parameters and applies
a physically-motivated perturbation to every spectrum. They are **stochastic**:
each `transform` call redraws, so calling `transform` K times yields K different
augmented batches (see `augmentation_workflow.md`). All take `random_state` for
reproducibility.

| type | class | key params | simulates |
|---|---|---|---|
| `add_noise` | `AddNoise` | `distribution` ∈ {`gaussian`, `poisson`, `exponential`}, `scale` | Detector / shot / read noise. `gaussian` for thermal/read noise; `poisson` for photon-count (shot) noise; `exponential` for skewed noise. `scale` sets magnitude. |
| `baseline_shift` | `BaselineShift` | `scale` | A random constant additive offset (baseline level drawn from a one-sided distribution) — day-to-day baseline drift. |
| `spectrum_scale` | `SpectrumScale` | `scale` | A random multiplicative gain (uniform, centered on 1) — overall intensity / path-length / concentration-independent scaling. |
| `gaussian_broadening` | `GaussianBroadening` | `sigma`, `mode`, `truncate` | Peak broadening via Gaussian convolution — resolution differences, thermal broadening. `sigma` in channels. |
| `index_shift` | `IndexShift` | `shift` (int), `padding_mode` | Integer x-axis shift drawn in `[-shift, +shift]` — wavenumber-calibration jitter. `padding_mode` fills exposed ends (`linear` default). |
| `fractional_shift` | `FractionalShift` | `shift` (float), `min_shift`, `padding_mode` | Sub-pixel x-axis shift via cubic-spline interpolation — fine calibration drift between measurements. |

## Choosing which to use

Augment for the variation your model will actually face in deployment:

```
Noisy detector / low signal              -> add_noise (match distribution to the noise type)
Baseline drifts run-to-run                -> baseline_shift
Intensity/gain varies (path length)       -> spectrum_scale
Instruments differ in resolution          -> gaussian_broadening
Wavenumber calibration jitters            -> index_shift (coarse) / fractional_shift (fine)
```

Combine several (a small amount of each) to cover multiple effects at once — the
`augment.py` spec chains them. Keep each magnitude realistic; see
`augmentation_workflow.md` for how to calibrate magnitudes with
`preview_augmentations.py`.

## Label-preserving

All of these keep the *reference value* (`y`) unchanged — a noisier or slightly
shifted spectrum of the same sample still has the same concentration/class. That
is why `augment.py` **tiles** the labels to match the expanded spectra. Do not
use augmentations that would change what `y` should be.
