# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

This is the **personal marketplace + plugin repository** for the `chemometrics`
Claude Code plugin — a suite of Agent Skills (plus slash commands and a subagent)
that guide a full chemometrics workflow on spectroscopy data (IR/NIR/Raman/ATR-FTIR/UV-Vis)
using [`chemotools`](https://pypi.org/project/chemotools/) + scikit-learn.

It is **not** the chemotools library. The library is a runtime dependency
(`pip install "chemotools[viz]"`); this repo only ships the skills/commands/agents that
drive it. The skills were originally authored in the chemotools repo under `skills/` and
migrated here to be released independently.

**Build plans and session handoff live in `notes/` — a local-only, git-ignored folder (not
published):** `notes/chemometrics_plugin_plan.md` (the full plan / source of truth for structural
decisions), `notes/preprocessing_skills_plan.md` (preprocessing rationale), and `notes/HANDOFF.md`
(what is done / what is next — read it first each session). These are working notes; if the folder
is absent (fresh clone), the repo is still complete and self-describing via this file + the READMEs.

## Layout

```
.claude-plugin/marketplace.json          # marketplace index (repo root = marketplace root)
plugins/chemometrics/                     # the plugin payload
├── .claude-plugin/plugin.json            # plugin manifest (bump version to ship updates)
├── skills/                               # each skill: SKILL.md + scripts/ references/ assets/
│   ├── spectral-preprocessing/           # preprocessing sub-hub
│   ├── baseline-scatter-correction/
│   ├── smoothing-derivatives/
│   ├── calibration-transfer/
│   ├── spectral-augmentation/
│   ├── chemometric-modeling/
│   ├── chemometric-validation/
│   ├── model-diagnostics/
│   ├── model-persistence/                # save/load: joblib + OpenModels JSON
│   └── chemometrics-workflow/            # orchestrator + scaffold_analysis.py
├── commands/                             # analyze.md, model.md, validate.md
├── agents/                               # chemometrician.md (read-only advisor)
├── tests/                                # smoke_test.py + check_frontmatter.py + check_manifests.py
└── README.md
notes/                                    # LOCAL-ONLY (git-ignored): build plans + HANDOFF
```

## Tooling & commands

No build system — this is a skills/docs repo. Scripts are driven by the installed library.

```bash
# one-time: install the runtime the scripts drive
pip install "chemotools[viz]" pandas        # or use a venv / uv

# run the suite's own tests (from repo root)
python plugins/chemometrics/tests/check_frontmatter.py   # names/descriptions/trigger hygiene
python plugins/chemometrics/tests/smoke_test.py          # exercises every script on synthetic data
# (or) pytest plugins/chemometrics/tests
```

Optional script deps, guarded with actionable messages: `matplotlib` (plots, via the `viz`
extra), `pandas` (CSV/Parquet I/O), `pyyaml` (YAML specs; JSON works without it),
`joblib` (persisting fitted pipelines).

## The skill-authoring pattern (imitate the existing five)

Each skill is a folder `skills/<name>/`:
- `SKILL.md` — YAML frontmatter (`name` = folder name, kebab-case; `description` ≤ 1024 chars,
  written as *trigger* text so Claude loads it on the right requests) + a **lean** body.
- `scripts/` — runnable, **importable** CLIs (the smoke test imports their core functions).
- `references/` — deep docs loaded on demand (progressive disclosure).
- `assets/` — decision-tree SVGs, templates.

## Conventions (inherited — see plan Part 4)

- **No leakage.** Every model/validation script fits on **train only**; CV runs on full
  Pipelines; use grouped folds (`GroupKFold`) for replicate/augmented spectra.
- **chemotools vs sklearn.** Never re-implement sklearn. Use sklearn for the ML engine
  (Pipeline, CV, GridSearchCV, PCA, metrics); use chemotools for the chemometrics-specific
  parts: `outliers` (applicability domain), `inspector`, `regression.PLSRegression`,
  `feature_selection` (VIP/SR). Wrap only chemometric aggregates (RMSEP/RPD/bias).
- **Current API names**; scripts importable *and* CLI-runnable; optional deps guarded.
- **Reproducible examples** on bundled datasets: `load_fermentation_train/test` (regression),
  `load_coffee` (3-class PLS-DA).
- **Progressive disclosure**: lean SKILL.md, depth in `references/`, decision SVGs in `assets/`.
- **Trigger hygiene**: the orchestrator's description is broad; each spoke's is narrow and
  non-overlapping. `check_frontmatter.py` guards this. (This is why the generic external
  `scikit-learn` skill was rejected — plan §1.4.)

## Portability requirement

A skill must work identically whether loaded from the plugin or copied to `~/.claude/skills/`.
**No skill may hardcode plugin paths or depend on plugin-only wiring** — scripts resolve
siblings by relative path only.

## GitHub handle

The plugin ships under the `Gnpd` GitHub account — repo `Gnpd/chemometrics-plugin`
(install: `/plugin marketplace add Gnpd/chemometrics-plugin`). Manifests and docs use the
canonical author/owner email `alejandro@g-npd.com`.
