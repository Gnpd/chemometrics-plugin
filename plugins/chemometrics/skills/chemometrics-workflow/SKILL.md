---
name: chemometrics-workflow
description: >-
  Run an end-to-end chemometrics analysis of spectroscopy data with chemotools + scikit-learn:
  explore, preprocess, model, validate, and deploy. Use this as the entry point when the user
  wants the whole workflow — "build a calibration/PLS model from these raw spectra", "analyze
  this spectral dataset end to end", "predict a property or class from spectra" — rather than a
  single step. Sequences the stages and routes each to the specialist skill: spectral-preprocessing,
  chemometric-modeling, chemometric-validation, and model-diagnostics.
---

# Chemometrics workflow (orchestrator)

The one **front door** for a whole-dataset request. This skill carries no
technique depth of its own — it establishes context, enforces the end-to-end
order and leakage discipline, and routes each decision to the specialist skill
that owns it.

## 1. Scope check — is this really end-to-end?

If the user wants a **single stage**, defer straight to that skill instead of
running the whole workflow:

- "which baseline / how do I smooth?" → **spectral-preprocessing** (+ spokes)
- "which model / how many latent variables?" → **chemometric-modeling**
- "cross-validate this / what metrics?" → **chemometric-validation**
- "is this spectrum safe to predict?" → **model-diagnostics**

Only run the full sequence for "analyze this dataset", "build a calibration model
from raw spectra", "predict a property/class end to end".

## 2. Establish context (before any fitting)

- **Modality** (IR/NIR/Raman/ATR-FTIR/UV-Vis) and **data shape / x-axis** — load
  with `spectral-preprocessing/scripts/load_spectra.py` (recovers the wavenumber
  axis, fixes 1-D shape).
- **Task** — regression (predict a number), classification (predict a class), or
  exploration (PCA, no target).
- **Replicates / augmentation** — are there multiple spectra per physical sample?
  If so, everything downstream needs **grouped** CV.
- **Split FIRST.** Hold out a test set (grouped/temporal if needed) before
  fitting anything. Every stage below is fit on **train only**.

## 3. The five stages → owning skill

| Stage | Intent | Owning skill |
|---|---|---|
| **Explore** | PCA scores/loadings; spot gross outliers & clusters | `chemometric-modeling` (PCA) + `model-diagnostics` |
| **Preprocess** | Correct / normalize / derivative → model-ready features | `spectral-preprocessing` (+ its spokes) |
| **Model** | Fit PLS / PLS-DA; choose latent variables; VIP/SR bands | `chemometric-modeling` |
| **Validate** | Leakage-free CV; RMSECV/RMSEP/R²/RPD/bias or acc/F1; permutation test | `chemometric-validation` |
| **Deploy / predict** | Persist the pipeline; guard every prediction with the applicability domain | `model-diagnostics` |

Route each hard choice to the owning skill; do not re-derive its logic here.

## 4. Order & leakage discipline (the invariant)

1. **Split first** — test set untouched until the end.
2. **One `Pipeline`** — preprocessing + model together, so CV refits preprocessing
   per fold (no leakage).
3. **Grouped CV** for replicate/augmented spectra (copies never straddle folds).
4. **Select LVs by CV**, never by training fit (1-SE rule).
5. **Fit the applicability domain on train**; AD-gate predictions at deploy time.
6. **Report on the held-out test set** (RMSEP / metrics), confirmed by a
   permutation test when `n` is small vs. the number of bands.

## 5. Scaffold a runnable starter

Emit an end-to-end script wiring all of the above, with `# TODO` markers naming
the spoke to consult for each choice:

```
python scripts/scaffold_analysis.py --dataset fermentation --task regression --out analysis.py
python scripts/scaffold_analysis.py --data spectra.csv --y y.csv --task classification --out run.py
```
The generated script uses only the public chemotools + scikit-learn API (no
plugin dependency), so it runs standalone and is safe to hand to a user.
`render()` is importable. Full narrative in `references/workflow_overview.md`;
the annotated stage map in `assets/workflow_map.svg`.

## Same brain, three surfaces

This SKILL.md is the single source of truth for the end-to-end procedure. It is
also exposed as the `/chemometrics:analyze` command and the `chemometrician`
subagent — both just follow this file.
