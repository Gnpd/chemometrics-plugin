# Applicability domain (AD)

## Why an AD at all

A calibration model interpolates within the space of its training spectra. Ask
it to predict a sample carrying a feature it never saw — a new interferent, a
contaminated probe, a different physical form, an instrument fault — and it will
still return a number, silently extrapolating. The applicability domain is the
guardrail: a quantitative boundary of "spectra like the ones I was trained on",
fit on the training model and checked at prediction time. Samples outside it get
flagged, not scored.

This is chemotools' `outliers` module (`HotellingT2`, `QResiduals`, `Leverage`,
`DModX`, `StudentizedResiduals`). scikit-learn has no equivalent — it is one of
the highest-value chemometrics-specific capabilities in the suite.

## The two core distances

A PCA/PLS model splits each spectrum into two parts: its projection **onto** the
model plane (the scores) and the part **left over** (the residual). Each has its
own notion of "too far".

| Statistic | Measures | Large value means | Catches |
|---|---|---|---|
| **Hotelling T²** | distance within the model plane (Mahalanobis on scores) | unusual *combination/magnitude* of scores, still explainable by the model | an extreme but in-model sample (very high/low concentration) |
| **Q-residuals** (SPE) | distance off the model plane (residual energy) | spectral structure the model *cannot* represent | a genuinely new kind of sample; new peak; artefact |

They are complementary: T² asks "is this an extreme version of what I know?", Q
asks "is this something I don't know at all?". Flag a sample outside **either**.
`applicability_domain.py` fits both and reports `outside_any`.

```python
from applicability_domain import build_ad_model, fit_domain, flag
model = build_ad_model(X_train, n_components=5)          # PCA of the training space
detectors = fit_domain(model, X_train, confidence=0.95)  # T2 + Q limits from train
result = flag(detectors, X_new)                          # per-sample in/out
safe = ~result["outside_any"]                            # gate predictions on this
```

You can pass a fitted PLS model or the full `Pipeline` instead of a PCA when you
want the AD defined in the *supervised* model's space.

## The other detectors

- **Leverage** — how much a sample pulls the model fit (diagonal of the hat
  matrix in score space). High-leverage *training* samples over-influence
  calibration; high-leverage *new* samples are extrapolations. Closely related to
  T².
- **DModX** — distance to the model in X, SIMCA-style; a per-sample residual
  distance often used for class modelling / one-class decisions.
- **StudentizedResiduals** — normalized regression residuals; large values are
  y-outliers (a sample the model predicts badly), useful for spotting bad
  reference values during calibration.

Rule of thumb: **T² + Q** for the general "safe to predict?" gate; add
**leverage** to find influential calibration points; **studentized residuals** to
audit the reference `y`; **DModX** for SIMCA-style class membership.

## Choosing the confidence

The `confidence` (default 0.95) sets each limit as a statistical threshold — at
0.95, about 5% of genuine in-domain samples may exceed it by chance (false
positives). Raise it (0.99) to flag only clear outliers and reduce false alarms;
lower it to be more cautious about extrapolation. Match it to the cost of a wrong
prediction vs. the cost of a rejected sample. The limits depend on the number of
components, so fit the AD with the **same model** (and LV count) you deploy.

## Discipline

- Fit the AD on **training data only** — it is part of the model, and fitting it
  on the samples you are judging defeats the purpose.
- Re-fit the AD whenever you re-fit or re-calibrate the model.
- The AD flags *extrapolation*, not *wrongness within domain* — a sample can be
  in-domain and still poorly predicted (that is what validation metrics and
  studentized residuals are for). Use the AD as a necessary, not sufficient,
  condition for trusting a prediction.
