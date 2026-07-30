# Selecting the number of latent variables (LVs)

The number of PLS latent variables (`n_components`) is the one hyperparameter
that governs the bias–variance trade-off of the model. Getting it right is the
difference between a model that generalizes and one that memorizes.

## Why not pick LVs by training fit

R² (or accuracy) on the training data is **monotone in LVs** — it only ever
improves as you add components, reaching 1.0 once the LVs span the data. So the
training fit tells you *nothing* about how many LVs to keep. Selecting LVs by
training R² is the single most common way to over-fit a PLS model.

Use **cross-validation** instead: the CV score (RMSECV / CV accuracy) improves,
bottoms out, then *worsens* as extra LVs start fitting fold-specific noise. That
turning point is the useful signal.

```
python scripts/select_components.py --task regression --dataset fermentation \
    --max-components 15 --plot lv_curve.png
```

## Reading the curve

- **Under-fit (too few LVs):** high RMSECV / low accuracy; both training and CV
  error are high and close. The model misses real structure.
- **Sweet spot:** CV error near its minimum; the gap to training error is
  moderate and stable.
- **Over-fit (too many LVs):** training error keeps dropping while CV error rises
  and the gap widens. The extra LVs model noise.

## The min rule vs. the 1-SE rule

`select_components.py` reports two optima:

- **min** — the LV count with the best mean CV score. Often a touch generous:
  CV scores are noisy, so the raw minimum can sit on a lucky fold split.
- **1-SE (recommended)** — the *most parsimonious* model whose mean CV score is
  within one standard error of the best. Breiman's one-standard-error rule: among
  models that are statistically indistinguishable from the best, take the
  simplest. Fewer LVs → more robust, more interpretable, less prone to over-fit.

The script computes the standard error of the mean across folds and applies the
rule automatically; the plot marks both choices.

## Practical notes

- **Grouped CV.** If replicate or augmented spectra of one physical sample exist,
  use `GroupKFold` so copies never straddle a fold — otherwise CV is
  optimistic and the chosen LV count is too high. See **chemometric-validation**.
- **Stratified CV** for classification keeps class proportions per fold.
- **Cap the sweep** sensibly: `max_components` is bounded by `min(n_features,
  n_samples − 1)`; on small datasets a handful of LVs is usually plenty.
- **Re-validate after band selection.** VIP/SR pruning changes the model; redo LV
  selection and CV on the reduced feature set.
- **Confirm on a held-out test set.** CV chooses LVs; a truly untouched test set
  (RMSEP) confirms the final model — that is the job of **chemometric-validation**.
