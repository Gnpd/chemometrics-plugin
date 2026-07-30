# Changelog

All notable changes to the `chemometrics` plugin. Versions follow
[Semantic Versioning](https://semver.org/); bump `version` in
`.claude-plugin/plugin.json` to ship an update to users.

## [Unreleased]

### Added
- **`model-persistence` skill** — save/load a fitted pipeline as exact `joblib` (default) or
  portable, pickle-free JSON via [OpenModels](https://github.com/Gnpd/openmodels). Script
  `persist_model.py` (`save_model`/`load_model`/`convert`/`check`/`inspect`, provenance
  `.meta.json` sidecar). The JSON path (optional `openmodels` dep) round-trips chemotools +
  scikit-learn pipelines via `chemotools.utils.discovery.all_estimators`. Producers/consumers
  are now format-aware: `fit_model.py --out … --format json`, and `applicability_domain.py`
  / `metrics.py` / `preprocess.py --save-model` load or write `.json` as well as `.joblib`.
- **Modeling & validation suite** completing the workflow (plan §3.3–3.6):
  - `chemometric-modeling` — PCA / PLS regression / PLS-DA (one-hot + argmax);
    latent-variable selection by CV (1-SE rule); VIP/SR band importance.
    Scripts: `fit_model.py`, `select_components.py`, `feature_importance.py`.
  - `chemometric-validation` — leakage-free `cross_validate` of the full pipeline
    (grouped / stratified / plain folds); RMSEP/R²/RPD/bias & accuracy/F1/confusion;
    y-scramble permutation test. Scripts: `cross_validate.py`, `permutation_test.py`,
    `metrics.py`.
  - `model-diagnostics` — applicability domain (Hotelling T² + Q-residuals) fit on
    train and applied to flag out-of-domain spectra; chemotools inspector
    diagnostics with the experimental `FutureWarning` suppressed. Scripts:
    `applicability_domain.py`, `diagnose_model.py`.
  - `chemometrics-workflow` — the single cross-domain orchestrator; routing-only
    SKILL.md + `scaffold_analysis.py`, which emits a runnable, plugin-independent
    end-to-end analysis script.
- **Commands**: `/chemometrics:analyze`, `/chemometrics:model`, `/chemometrics:validate`.
- **Subagent**: `chemometrician` (read-only end-to-end advisor).
- **Tests**: `check_manifests.py` (manifest + layout + portability lint);
  trigger-overlap check added to `check_frontmatter.py`; smoke tests extended to
  every new script (modeling fit/select/VIP-SR, grouped-CV leakage guard,
  permutation p-value, applicability-domain out-of-domain flag, inspector build,
  scaffold render/compile).

## [0.1.0]

### Added
- Initial migration of the preprocessing suite from the `chemotools` repo:
  `spectral-preprocessing` (hub) + `baseline-scatter-correction`,
  `smoothing-derivatives`, `calibration-transfer`, `spectral-augmentation`.
- Plugin + marketplace manifests; `check_frontmatter.py` and `smoke_test.py`.
