---
name: smoothing-derivatives
description: >-
  Denoise spectra and compute spectral derivatives with chemotools. Use when choosing
  smoothing (Savitzky-Golay, Whittaker, mean/median/modified-sinc) or derivative
  (Savitzky-Golay, Norris-Williams) settings — window length, polynomial order, derivative
  order — and balancing noise reduction against peak and resolution preservation, including
  despiking Raman cosmic rays and using derivatives to remove baselines.
---

# Smoothing & derivatives

Reduce noise and/or take derivatives without destroying the peaks that carry the
chemistry. A spoke of the **spectral-preprocessing** hub — return there for the
overall pipeline and order.

## The core tradeoff

Every smoothing/derivative choice trades **noise reduction** against
**resolution**. More smoothing removes more noise but flattens and broadens
peaks; higher derivative orders resolve overlapping bands but amplify noise.
Always use the *minimum* that meets the need — never "smoother to be safe."

`assets/smoothing_derivative_decision.svg` summarizes the choices.

## Smoothing only

Read `references/smoothing_methods.md`. Short version:

- **Default** → `SavitzkyGolayFilter` (preserves peaks). Set `window_length`
  (odd, narrower than your narrowest peak) and `polyorder` (≈2).
- **One continuous smoothness dial** → `WhittakerSmooth` (`lam`).
- **Raman cosmic-ray spikes** → `MedianFilter` **first**, before baseline.
- **Frequency-selective, sharp cutoff** → `ModifiedSincFilter`.

Tune the window:
```
python scripts/tune_smoothing.py --input spectra.csv --polyorder 2
```
Pick the smallest window with adequate `noise_reduction` and `peak_retention`
near 1.0.

## Derivatives

Read `references/derivative_methods.md`. Short version:

- **Use `SavitzkyGolay`** — it smooths and differentiates in one pass (preferred
  over smoothing then differentiating separately).
- `deriv=1` removes a constant offset and sharpens; `deriv=2` also removes a
  linear baseline and resolves overlapping bands (more noise → larger window).
- `polyorder` must be **greater than** `deriv`.
- Because a derivative removes additive/linear baselines, a derivative-based
  pipeline often needs **no separate baseline step**.
- Derivative output has real negative lobes — do **not** `NonNegative`-clip it.

Preview orders side by side:
```
python scripts/derivative_preview.py --input spectra.csv --window 21 --polyorder 2 --out deriv.png
```

## Order (reminder)

`smoothing`/`derivative` go after baseline and scatter, before scaling. The
Savitzky-Golay derivative typically replaces a separate smoothing step. Full
rationale in the hub's `references/ordering_guide.md`.

## Scripts

| script | purpose |
|---|---|
| `scripts/tune_smoothing.py` | Sweep Savitzky-Golay window; report noise reduction vs. peak retention. |
| `scripts/derivative_preview.py` | Show `deriv` 0/1/2 with chosen window/polyorder; report noise; optional figure. |
