---
name: spectral-augmentation
description: >-
  Augment spectroscopy training data with chemotools to build more robust PCA/PLS models,
  especially with few samples. Use when the user wants to expand a training set with
  synthetic-but-realistic spectral variants — adding noise, baseline shifts, intensity
  scaling, peak broadening, or wavenumber shifts — to improve generalization and robustness
  to instrument/sample variation. Covers AddNoise, BaselineShift, SpectrumScale,
  GaussianBroadening, IndexShift, FractionalShift, and how to apply augmentation without leakage.
---

# Spectral augmentation

Expand a **training** set with plausible synthetic variants so models generalize
better — most useful when samples are scarce (common in spectroscopy) or the
model must tolerate instrument/sample variation. A spoke of the
**spectral-preprocessing** hub. This serves model *training*, not spectral
*cleaning* — use it alongside, not instead of, preprocessing.

## The rules that matter most

Getting the mechanics right matters more than which transformer you pick
(`references/augmentation_workflow.md`, summarized in
`assets/augmentation_decision.svg`):

1. **Augment TRAIN only** — never validation/test. Split first, expand the
   training side.
2. **Keep all copies of a sample in one CV fold** (`GroupKFold`) — a copy in
   validation while its original is in train is leakage.
3. **Augment before preprocessing** — augmentations model raw-measurement
   variation; preprocess + fit on the expanded set, apply preprocessing
   unchanged to untouched test.
4. **It's a data step, not a Pipeline step** — augmentation expands N→K·N, which
   a scikit-learn `Pipeline` can't do; `scripts/augment.py` handles the expansion
   and tiles labels (augmentations are label-preserving).
5. **Match magnitude to reality** — verify with `scripts/preview_augmentations.py`;
   too much teaches noise, too little does nothing.
6. **Seed** for reproducible expansions.

## Choosing what to simulate

Augment for the variation your model will actually face
(`references/augmentation_methods.md`):

| Real-world variation | Transformer |
|---|---|
| Detector / shot noise | `AddNoise` (`distribution` = gaussian/poisson/exponential) |
| Run-to-run baseline drift | `BaselineShift` |
| Intensity / gain / path-length changes | `SpectrumScale` |
| Resolution differences between instruments | `GaussianBroadening` (`sigma`) |
| Coarse wavenumber-calibration jitter | `IndexShift` |
| Sub-pixel calibration drift | `FractionalShift` |

Combine a small amount of several rather than one large effect.

## Do it

Calibrate magnitudes first:
```
python scripts/preview_augmentations.py --input train.csv --scale 0.02 --shift 3 --out aug.png
```
Aim for a visible change with correlation-to-original clearly below 1.0 but not
collapsed.

Then expand the training set from a spec (chained augmentations):
```
python scripts/augment.py --input train.csv --labels train_y.csv \
    --spec aug_spec.yaml --copies 5 --seed 0 \
    --out train_aug.csv --out-labels train_aug_y.csv
```
Fit your preprocessing + model on `train_aug.*`; evaluate on the untouched
validation/test set. Augmentation should help or not hurt validation — if it
hurts, dial back the magnitudes.

## Scripts

| script | purpose |
|---|---|
| `scripts/preview_augmentations.py` | Apply each augmentation to sample spectra; report change/correlation; optional overlay figure. |
| `scripts/augment.py` | Expand a training set (original + K augmented batches) from a spec; tile labels; reproducible via `--seed`. |
