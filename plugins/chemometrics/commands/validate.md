---
description: Validate a chemometric model — leakage-free cross-validation, chemometric metrics, and a permutation significance test.
argument-hint: "[dataset name or path] [regression|classification] [--groups groupfile]"
---

Validate a model rigorously by following the **chemometric-validation** skill.
Load that skill and apply its guidance.

Target: **$ARGUMENTS**

Do:

1. Confirm the fold strategy: **grouped** (`GroupKFold`) if replicate or augmented
   spectra of one physical sample exist — this is mandatory to avoid leakage —
   otherwise stratified (classification) or plain `KFold` (regression).
2. Cross-validate the **whole** pipeline (preprocess + model together) so every
   fold refits preprocessing —
   `skills/chemometric-validation/scripts/cross_validate.py`. Report each metric
   as `mean ± std`.
3. Report the chemometric metric set — RMSEP/R²/RPD/bias (regression) or
   accuracy/F1/confusion (classification) — via `metrics.py`; never a single number.
4. When `n` is small vs. the number of bands, run the y-scramble
   **permutation test** (`permutation_test.py`) and report the p-value.
5. Flag over/under-fitting from the component curve and the train-vs-CV gap.

Report the fold strategy used and why, the metrics with spread, and the
significance result. If a grouped structure was ignored, say so and re-run grouped.
