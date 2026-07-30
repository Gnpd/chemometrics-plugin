# chemometrics-plugin

A Claude Code **marketplace** hosting the **`chemometrics`** plugin — Agent Skills, slash commands, and a subagent that guide a full chemometrics workflow on spectroscopy data (IR / NIR / Raman / ATR-FTIR / UV-Vis) with [`chemotools`](https://pypi.org/project/chemotools/) and [`scikit-learn`](https://pypi.org/project/scikit-learn/).

The plugin itself lives at [`plugins/chemometrics/`](plugins/chemometrics/). This repo root is the marketplace ([`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json)).

## Install

```
/plugin marketplace add Gnpd/chemometrics-plugin
/plugin install chemometrics@chemometrics-marketplace
/plugin list
```

Then install the runtime the scripts drive:

```bash
pip install "chemotools[viz]" pandas
```

Skills auto-activate on matching requests (e.g. "build a PLS calibration from these NIR spectra"); commands appear as `/chemometrics:analyze`, etc.

**Standalone use (no plugin):** copy any `plugins/chemometrics/skills/<name>/` folder into `~/.claude/skills/` — it works fully, minus the `/chemometrics:` namespace.

## What's included

- **10 skills** — 5 preprocessing (`spectral-preprocessing` hub + `baseline-scatter-correction`, `smoothing-derivatives`, `calibration-transfer`, `spectral-augmentation`), plus `chemometric-modeling`, `chemometric-validation`, `model-diagnostics`, `model-persistence` (save/load models as joblib or portable JSON), and the `chemometrics-workflow` orchestrator.
- **3 commands** — `/chemometrics:analyze`, `/chemometrics:model`, `/chemometrics:validate`.
- **1 subagent** — `chemometrician`, a read-only end-to-end analyst.

All scripts are exercised by the test suite (`plugins/chemometrics/tests/`).

## License

MIT.
