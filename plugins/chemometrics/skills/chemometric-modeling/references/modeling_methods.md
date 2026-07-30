# Modeling methods — PCA vs PLS vs PLS-DA

## Why PLS for spectra

Spectroscopic data is **wide and collinear**: hundreds–thousands of bands, far
more than samples, and neighbouring bands move together. Ordinary least squares
is undefined (p ≫ n) and ridge/lasso ignore the covariance with the target.
**Partial Least Squares (PLS)** projects X onto a few **latent variables (LVs)**
chosen to maximize covariance with **y**, then regresses on those. This is why
PLS — not linear/logistic regression on raw bands — is the chemometrics default.

| Method | Supervised? | Output | Use when |
|---|---|---|---|
| **PCA** | no | scores/loadings (variance) | Explore structure, spot outliers/clusters, before you have (or trust) a target. |
| **PLS regression** | yes | continuous ŷ | Quantify a property: concentration, moisture, octane number. |
| **PLS-DA** | yes | class label | Categorical outcome: origin, grade, adulterated vs. authentic. |

PCA finds directions of maximum **variance** in X alone; PLS finds directions of
maximum **covariance** with y. Use PCA to look, PLS to predict.

## PLS regression

`chemotools.regression.PLSRegression` subclasses sklearn's `_PLS`, so it is
API-identical (`fit`/`predict`/`transform`) and drops into a `Pipeline` and
`cross_val_score`; it adds doc-link/HTML niceties. sklearn's
`cross_decomposition.PLSRegression` is an equally valid substitute.

- `n_components` (LVs) is *the* hyperparameter — choose it by CV
  (`select_components.py`, `component_selection.md`), never by training fit.
- `scale=True` standardizes each column; keep it on unless you have a reason not
  to (e.g. you already scaled in the preprocessing head — then it is harmless but
  redundant).

## PLS-DA — classification with PLS

There is no dedicated chemotools/sklearn "PLS-DA" class; build it from PLS.

### Default recipe (this plugin): PLS on one-hot + argmax

Implemented as `PLSDA` in `scripts/fit_model.py`:

1. One-hot encode the class labels (`LabelBinarizer`); binary problems get two
   columns so `argmax` is well defined.
2. Fit a single multi-output `PLSRegression` on the one-hot target.
3. Predict by taking the class with the largest PLS output (`argmax`).

Chosen as the default because it is the closest analogue to classic PLS-DA, is a
single estimator (one `n_components` to tune, one object to persist), and slots
straight into a `Pipeline`/CV. `PLSDA` also exposes `predict_proba` (a softmax
over the PLS scores) as a convenience — treat it as a ranking, not a calibrated
probability.

### Alternative: LDA on PLS scores

Use PLS (or `PLSRegression.transform`) purely for dimension reduction, then fit
`LinearDiscriminantAnalysis` on the LV scores:

```python
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# PLS as a supervised dimensionality reducer, LDA as the decision rule
clf = Pipeline([("pls", PLSRegression(n_components=10)),
                ("lda", LinearDiscriminantAnalysis())])
```
More principled decision boundaries (LDA models within/between-class
covariance), at the cost of a two-stage model with two things to tune. Prefer it
when classes overlap and the argmax rule misclassifies near the boundary.

Both recipes need **stratified** (and, for replicates, **grouped**) folds and the
same LV-by-CV discipline as regression.

## Preprocessing belongs upstream

Baseline/scatter/smoothing/derivative/scaling choices live in
**spectral-preprocessing** and its spokes. Wire the chosen chain as the head of
the same `Pipeline` (`fit_model.py --spec`), so it is fit on train only. A
derivative or SNV step often does more for a PLS model than an extra LV.
