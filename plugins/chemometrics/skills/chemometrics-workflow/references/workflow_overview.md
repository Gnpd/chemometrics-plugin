# End-to-end chemometrics workflow

The five stages, the decision points, and the leakage rules in one place. The
orchestrator sequences these and routes each to its specialist skill; this
document is the narrative behind the routing table.

## The pipeline of stages

```
raw spectra
   │
   ├─ 0. Load & shape ............. load_spectra.py  (x-axis, (n_samples, n_features))
   │
   ├─ 0. SPLIT FIRST ............. train / test  (grouped or temporal if needed)
   │
   ├─ 1. Explore ................. PCA scores/loadings; gross outliers, clusters
   │        owner: chemometric-modeling (PCA) + model-diagnostics
   │
   ├─ 2. Preprocess .............. baseline / scatter / smooth / derivative / scale
   │        owner: spectral-preprocessing (+ spokes)          [Pipeline HEAD]
   │
   ├─ 3. Model ................... PLS / PLS-DA; select LVs by CV; VIP/SR bands
   │        owner: chemometric-modeling                        [Pipeline TAIL]
   │
   ├─ 4. Validate ................ leakage-free CV; RMSEP/R²/RPD/bias or acc/F1;
   │        owner: chemometric-validation                       permutation test
   │
   └─ 5. Deploy / predict ........ persist pipeline; applicability-domain gate
            owner: model-diagnostics
```

Stages 2 and 3 are the **head and tail of one `Pipeline`**. That is the whole
leakage story: because preprocessing and model live in one estimator, every CV
fold and every refit re-learns preprocessing on its own training rows.

## Decision points (routed to the owning skill)

1. **Task?** regression / classification / exploration → picks the model
   (chemometric-modeling) and the metrics & CV scheme (chemometric-validation).
2. **Replicates or augmented copies?** → grouped CV throughout
   (chemometric-validation); split by group up front.
3. **Which preprocessing?** modality- and problem-specific → spectral-preprocessing
   and its spokes (baseline-scatter-correction, smoothing-derivatives,
   calibration-transfer if multi-instrument, spectral-augmentation if data-poor).
4. **How many latent variables?** CV + 1-SE rule → chemometric-modeling.
5. **Is the signal real?** permutation test when bands ≫ samples →
   chemometric-validation.
6. **Safe to predict a given new spectrum?** applicability domain →
   model-diagnostics.

## The leakage rules (non-negotiable)

- **Split before you fit anything.** The test set does not exist during
  preprocessing choice, LV selection, or AD fitting.
- **Fit on train only** — preprocessing statistics, PLS loadings, scaler means,
  band selection, and the applicability domain are all learned from train.
- **One Pipeline, cross-validated whole.** Never preprocess the full matrix and
  then CV the model — that leaks the test set into the preprocessing.
- **Grouped folds for copies.** Replicate/augmented spectra of one sample must
  stay within a single fold, or CV is optimistic and LV choice inflated.
- **AD is part of the model.** Fit it on train; apply it to new samples at
  prediction time; re-fit it whenever the model is re-calibrated.

## Using the scaffold

`scaffold_analysis.py` writes a script that already obeys every rule above:
split → one Pipeline (SNV + Savitzky-Golay derivative + PLS/PLS-DA) → LV
selection by CV (1-SE) → fit on train → held-out metrics → AD fit on train →
AD-guarded prediction. Each `# TODO` names the spoke to consult to refine that
choice. Treat the scaffold as a correct-by-construction starting point, then
deepen each stage with its specialist skill — do not hand-wire the order from
scratch.
