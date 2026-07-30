# Augmentation workflow

Augmentation expands a **training** set with synthetic-but-plausible variants so
a model generalizes better — especially valuable with few samples (common in
spectroscopy) or when the model must be robust to instrument/sample variation.
Done wrong it leaks or teaches noise. The rules below matter more than the choice
of transformer.

## Rule 1 — augment TRAIN only, never validation/test

Test/validation spectra must stay untouched so they estimate real-world
performance. Augmenting them inflates scores meaninglessly. Split first, then
augment only the training partition.

## Rule 2 — keep augmented copies of a sample together in CV

An augmented copy is nearly identical to its original. If a copy lands in the
validation fold while its original is in the training fold, the model has
effectively seen the answer — leakage. When cross-validating, **group by the
original sample** (e.g. `sklearn.model_selection.GroupKFold`) so all copies of a
spectrum share a fold. Simplest safe pattern: do the train/validation split
*before* augmenting, and only expand the training side.

## Rule 3 — augmentation comes BEFORE preprocessing

Augmentations simulate variation in the **raw measurement** (noise, baseline,
calibration jitter). Apply them to raw-ish training spectra, then fit your
preprocessing pipeline + model on the expanded set. Conceptually:

```
raw train  --augment(xK)-->  expanded raw train  --preprocess+fit-->  model
raw test   ------------------------------------->  preprocess(transform) -> predict
```

The preprocessing pipeline is still fit on the (expanded) training data only, and
applied unchanged to test — the hub's leakage rules still hold.

## Rule 4 — augmentation is a data step, not a Pipeline step

A scikit-learn `Pipeline` `transform` maps N samples to N samples; it cannot
expand N to K·N. So augmentation is a **dataset-preparation** step you run
*before* building the modeling pipeline — that is what `scripts/augment.py` does
(stack original + K augmented batches, tile labels). Don't drop an augmentation
transformer into your modeling pipeline expecting more rows.

## Rule 5 — match magnitude to reality; verify

Augment for variation the model will actually meet. Too little does nothing; too
much (e.g. a shift larger than any real calibration drift, noise louder than the
instrument's) teaches the model to ignore real signal. Calibrate magnitudes with
`scripts/preview_augmentations.py`: aim for a **visible but high-correlation**
change (corr to original clearly below 1.0 but not collapsed). Combine a small
amount of several effects rather than one large one.

## Rule 6 — seed for reproducibility

Pass a fixed `--seed`. chemotools augmenters advance their RNG across `transform`
calls, so a seed makes the *entire* K-copy expansion deterministic — two runs
with the same seed and spec produce identical training sets, which you want for
comparable experiments.

## Putting it together

```python
# 1. split first
X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state=0)

# 2. augment TRAIN only  (scripts/augment.py does this from a spec)
X_tr_aug = augment(X_tr, spec, copies=5, seed=0)
y_tr_aug = np.tile(y_tr, 6)              # original + 5 copies

# 3. preprocess + model, fit on augmented TRAIN, evaluate on untouched VAL
pipe.fit(X_tr_aug, y_tr_aug)
score = pipe.score(X_val, y_val)         # honest estimate
```

Augmentation should *help or not hurt* validation performance. If it hurts,
your magnitudes are unrealistic or you're augmenting effects the model shouldn't
be invariant to — dial back and re-check with `preview_augmentations.py`.
