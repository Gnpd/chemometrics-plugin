# Adaptation / calibration-transfer methods

Two distinct problems live here. Diagnose which one you have first.

1. **X-axis misalignment** — spectra sit on *different grids* (wavenumber /
   wavelength arrays differ). You cannot even stack them into one matrix.
   → `XAxisInterpolator`.
2. **Instrument response difference** — spectra share a grid but a model trained
   on instrument A performs poorly on instrument B because B's response (gain,
   resolution, baseline) differs. → `DirectStandardization` /
   `PiecewiseDirectStandardization`.

All in `chemotools.adaptation`.

## XAxisInterpolator — resample onto a common grid

Defines a `common_x_axis` once at construction and, at each `transform`,
resamples every spectrum from its own `x_axis` (passed as metadata) onto that
grid.

- `common_x_axis`: the shared grid (e.g. `np.linspace(650, 1550, 1000)`).
- `method`: `'linear'` (fast, smooth data), `'cubic'` (smooth all-rounder),
  `'pchip'` (monotonic, avoids overshoot near peaks).
- `left` / `right`: fill values for points outside an input grid. **They default
  to NaN** — set to `0` (or another sentinel) if downstream steps can't handle
  NaN, or choose a `common_x_axis` fully inside every input grid. This is the
  #1 gotcha (see hub `pitfalls.md` §4).
- Uses metadata routing (`references/metadata_routing.md`); `x_axis` may be
  shared `(n_features,)` or per-sample `(n_samples, n_features)`.

`scripts/align_axes.py` wraps this end to end.

## DirectStandardization (DS)

Learns a single global linear map between two instruments from **transfer
standards** — samples measured on both. Fit with the target-instrument spectra
as `X` and the source-instrument spectra as `X_source`:

```python
ds = DirectStandardization().fit(X_target_std, X_source=X_source_std)
X_mapped = ds.transform(X_target_new)   # target spectra, expressed in source space
```

Good when the instrument difference is roughly uniform across the spectrum. With
no `X_source` it degrades to the identity. Stored compactly (low-rank) rather
than as a full `n_features × n_features` matrix.

## PiecewiseDirectStandardization (PDS)

Like DS but learns **local** maps in a sliding window (windowed PLS), so it
adapts to instrument differences that vary across the axis (resolution changes,
wavelength-dependent gain).

- `window_length` (default 25): channels per local model. Larger = smoother,
  more context; smaller = more local flexibility.
- `n_components`, `scale`: the local PLS settings.

PDS usually transfers better than DS when the difference is
wavelength-dependent, at the cost of more parameters. Compare both with
`scripts/transfer_demo.py`.

## Choosing

```
Different x-axis grids                          -> XAxisInterpolator (align first)
Uniform instrument response difference           -> DirectStandardization
Wavelength-dependent response difference          -> PiecewiseDirectStandardization
```

Alignment (if needed) comes **before** standardization, and both come early in
the pipeline — everything downstream assumes a common grid and instrument.

## Related: removing external-parameter variance

If the nuisance variation comes from a measured external parameter (temperature,
humidity) rather than an instrument swap, see `chemotools.projection`
(`ExternalParameterOrthogonalization`, `OrthogonalSignalCorrection`) via the hub.
