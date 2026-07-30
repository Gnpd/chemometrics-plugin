# chemometrics (Claude Code plugin)

Agent Skills + slash commands + a subagent for an end-to-end chemometrics workflow on
spectroscopy data, built on [`chemotools`](https://pypi.org/project/chemotools/) + [`scikit-learn`](https://pypi.org/project/scikit-learn/).

## Skills

### Preprocessing (shipped)

| Skill | Use it for |
|---|---|
| **`spectral-preprocessing`** (sub-hub) | The preprocessing workflow: load/shape data, choose techniques and order, assemble a leakage-free `Pipeline`, diagnose. Routes hard choices to the spokes. |
| **`baseline-scatter-correction`** | Baseline (AirPLS/ArPLS/AsLS/polynomial/rubberband) and scatter (SNV/RNV/MSC/EMSC) correction. |
| **`smoothing-derivatives`** | Smoothing (Savitzky-Golay/Whittaker) and derivatives; noise-vs-resolution tuning; Raman despiking. |
| **`calibration-transfer`** | Different x-axis grids, multi-instrument transfer (DS/PDS), per-measurement metadata via sklearn routing. |
| **`spectral-augmentation`** | Expand a training set with synthetic-but-realistic variants for more robust models. |

### Modeling & validation (shipped)

| Skill | Use it for |
|---|---|
| **`chemometric-modeling`** | Choose & fit a model: PCA, PLS regression, PLS-DA; select latent variables by CV; VIP/SR band selection. |
| **`chemometric-validation`** | Leakage-free CV (grouped folds), RMSECV/RMSEP/R²/RPD/bias or classification metrics, permutation test, over/underfit checks. |
| **`model-diagnostics`** | Applicability domain (Hotelling T²/Q-residuals/leverage/DModX) and inspector-based visual diagnostics. |
| **`model-persistence`** | Save/load a fitted pipeline as exact joblib or portable pickle-free JSON ([OpenModels](https://github.com/Gnpd/openmodels)); convert/inspect; provenance sidecar. |
| **`chemometrics-workflow`** (orchestrator) | Front door for whole-dataset requests: explore → preprocess → model → validate → deploy; routes each stage to its spoke. |

Each skill folder holds `SKILL.md` + `scripts/` (runnable, importable CLIs) + `references/`
(deep docs) + `assets/` (decision SVGs, templates).

## Commands & subagent (shipped)

- **`/chemometrics:analyze`** — run an end-to-end analysis of a dataset.
- **`/chemometrics:model`** — fit a model and choose latent variables by CV.
- **`/chemometrics:validate`** — cross-validate, report metrics, permutation test.
- **`chemometrician`** — delegated end-to-end analyst subagent (read-only advisor:
  reports + a runnable script; does not modify your files unless asked).

## Install

```
/plugin marketplace add Gnpd/chemometrics-plugin
/plugin install chemometrics@chemometrics-marketplace
/plugin list
```
Standalone (no plugin): copy any `skills/<name>` folder into `~/.claude/skills/`
(cross-skill scripts resolve their siblings by relative path, so copy the sibling
skills a script uses too — e.g. `chemometric-*` reuse `spectral-preprocessing`).

Runtime: `pip install "chemotools[viz]" pandas` (matplotlib/pandas/pyyaml/joblib
are optional and guarded with actionable messages). For portable JSON model export,
`pip install openmodels` (used only by `model-persistence`'s `.json` path).

## Tests

```bash
pip install "chemotools[viz]" pandas
python tests/check_frontmatter.py   # names / descriptions / trigger hygiene (10 skills)
python tests/check_manifests.py     # plugin + marketplace manifests, layout, portability
python tests/smoke_test.py          # exercises every script on synthetic spectra
# or: pytest tests
```
