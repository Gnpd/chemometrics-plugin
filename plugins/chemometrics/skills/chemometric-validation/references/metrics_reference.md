# Metrics reference

## Regression

Let `y` be the reference values, `ŷ` the predictions, `n` the sample count.

| Metric | Formula | Meaning |
|---|---|---|
| **RMSEC** | RMSE on the training (calibration) set | Optimistic; over-fits with more LVs. |
| **RMSECV** | RMSE from cross-validation | Generalization estimate used to choose LVs. |
| **RMSEP** | RMSE on a held-out test/prediction set | The number you report. Same units as `y`. |
| **R²** | `1 − SS_res/SS_tot` | Fraction of variance explained. Beware: high R² with high RMSEP means a narrow reference range. |
| **RPD** | `std(y) / RMSEP` | Units-free quality index (see ranges below). |
| **RPIQ** | `IQR(y) / RMSEP` | RPD variant robust to non-normal `y`. |
| **bias** | `mean(ŷ − y)` | Systematic offset. Should be ≈ 0; persistent bias = drift/transfer problem. |
| **SEP** | std of `(ŷ − y)` about their mean | Bias-corrected RMSEP: `RMSEP² ≈ bias² + SEP²`. |

`metrics.py` reports RMSEP, R², RPD, bias, and SEP.

### RPD interpretation (rule of thumb — application-dependent)

| RPD | Reading |
|---|---|
| < 1.5 | Poor; not usable for prediction. |
| 1.5 – 2.0 | Rough screening only. |
| 2.0 – 2.5 | Approximate quantitative prediction. |
| 2.5 – 3.0 | Good. |
| > 3.0 | Excellent. |

These bands are conventional, not universal — a hard property (low `std(y)`, high
reference-method noise) may be useful at a lower RPD, and some regulated
applications demand more. Always report RMSEP in the property's units alongside
RPD so the reader can judge against the actual requirement.

### Why not a single number

R² alone hides bias and range effects; RMSEP alone is unitless-context-free
without `std(y)`. Report the set: RMSEP (magnitude), R²/RPD (relative quality),
bias (systematic error). A model with R² = 0.95 but a large bias is broken for
quantitation even though it "looks" accurate.

## Classification

| Metric | Meaning |
|---|---|
| **accuracy** | Fraction correct. Misleading under class imbalance. |
| **macro-F1** | Unweighted mean of per-class F1; treats rare classes equally. |
| **confusion matrix** | Which classes are confused with which — always inspect it. |
| **balanced accuracy** | Mean per-class recall; robust to imbalance (sklearn). |

For imbalanced problems prefer macro-F1 / balanced accuracy over raw accuracy,
and read the confusion matrix to see *where* errors fall. `metrics.py` reports
accuracy, macro-F1, and the confusion matrix; add sklearn's
`classification_report` for per-class precision/recall.

## Significance — permutation test

A model can achieve high R²/accuracy on wide spectral data **by chance**. The
y-scramble permutation test (`permutation_test.py`) refits on shuffled labels to
build the null distribution of the CV score; the p-value is the fraction of
permutations that match or beat the true model. p < 0.05 → the signal is real.
Report it whenever `n` is small relative to the number of bands.
