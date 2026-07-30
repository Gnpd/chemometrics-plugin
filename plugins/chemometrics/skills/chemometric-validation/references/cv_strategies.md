# Cross-validation strategies (leakage-free)

## Why CV the whole pipeline

Every preprocessing step that *learns* from data — SNV/MSC references, scaler
means and standard deviations, PCA/PLS loadings, band selection — must be fit on
the **training rows of each fold only**. If you preprocess the full matrix once
and then cross-validate the model, the test rows have already influenced the
preprocessing, and the CV score is optimistically biased.

The fix is mechanical: put preprocessing and model in one `Pipeline` and pass
*that* to `cross_validate`. scikit-learn then refits the entire chain per fold.
`cross_validate.py` and `permutation_test.py` do exactly this.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GroupKFold
pipe = Pipeline([("snv", ...), ("deriv", ...), ("pls", ...)])   # preprocess + model
cross_validate(pipe, X_train, y_train, cv=GroupKFold(5), groups=groups)  # refits per fold
```

## Choosing folds

| Situation | Splitter | Why |
|---|---|---|
| Replicate / augmented spectra of one sample | **`GroupKFold`** | Copies of one physical sample must all be in the same fold, or the model "sees" a near-duplicate of each test spectrum during training. Grouped CV is the honest estimate. |
| Classification | **`StratifiedKFold`** | Preserves class proportions in every fold; essential with imbalanced classes. |
| Plain regression, independent samples | **`KFold`** (shuffled) | No grouping needed. |
| Time / batch structure | group or block by batch | Never let a future batch predict its own past via shared preprocessing. |

**The grouped-CV trap.** Data augmentation (spectral-augmentation skill) and lab
replicates create multiple rows per sample. Ungrouped CV then places near-copies
on both sides of the split and reports a score the model will never achieve on
genuinely new samples. Always pass `--groups` when copies exist. A quick check:
grouped CV scores noticeably *worse* than ungrouped is the leakage you just
removed, not a regression.

## Train / validation / test

- **Cross-validation** (on the training set) chooses hyperparameters — number of
  LVs, preprocessing options.
- A **held-out test set**, untouched until the end, gives the final RMSEP /
  accuracy you report. CV estimates generalization; the test set confirms it.
- Split **first**, before any fitting. `train_test_split` (or a grouped/temporal
  split) up front; everything downstream is fit on train only.

## Nested CV when you also tune

If you select hyperparameters *and* estimate performance from CV, a single loop
leaks the tuning choice into the score. Use **nested CV**: an inner loop
(`GridSearchCV`) tunes, an outer loop scores the tuned procedure.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
inner, outer = GroupKFold(4), GroupKFold(5)
search = GridSearchCV(pipe, {"pls__n_components": range(1, 15)}, cv=inner,
                      scoring="neg_root_mean_squared_error")
scores = cross_val_score(search, X, y, groups=groups, cv=outer)  # honest, tuned estimate
```
For a single dominant hyperparameter (LV count) on small data, a component curve
plus a held-out test set is often enough; reach for nested CV when tuning several
knobs or when the dataset is large enough to afford it.

## Reading the numbers

- Report `mean ± std` across folds. A large std means the estimate is unstable —
  usually too few samples per fold or hidden group structure.
- Compare **RMSECV** (from CV) with **RMSEP** (held-out test). RMSEP ≫ RMSECV
  signals over-fitting to the CV procedure or a distribution shift in the test set
  (check the applicability domain — model-diagnostics).
- A non-zero **bias** that persists across folds is systematic: instrument drift,
  a calibration-transfer gap, or a mislabeled reference method.
