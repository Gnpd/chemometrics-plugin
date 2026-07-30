---
name: model-persistence
description: >-
  Save, load, and share a fitted chemometric pipeline. Use when persisting a fitted
  scikit-learn/chemotools pipeline to disk, exporting it to portable pickle-free JSON with
  OpenModels for sharing, deployment, or archiving, reloading a saved model to predict, converting
  between joblib and JSON, or recording provenance (library versions, latent variables, x-axis)
  beside the artifact. Covers the joblib-vs-JSON trade-off and safe loading. Delegates fitting to
  chemometric-modeling and prediction-time safety to model-diagnostics.
---

# Model persistence

Once a pipeline is fit, this skill owns getting it **off the notebook** — to disk, to a
teammate, to production — and back again intact. It covers the **Deploy** step of the
workflow (persist the pipeline); fitting belongs to **chemometric-modeling** and the
applicability-domain gate to **model-diagnostics**.

Persist the *whole* `Pipeline` (preprocessing head + model), never the bare estimator — the
saved object must reproduce the exact same transform on new spectra.

## Two formats — pick by where the model is going

Decide from `assets/persistence_decision.svg`:

- **`joblib` (`.joblib`) — the default.** Exact, fast, dependency-light. Best for reusing a
  model **locally in the same environment**. It is a pickle: only load files you trust, and it
  can break across scikit-learn / chemotools versions.
- **OpenModels JSON (`.json`) — for leaving the machine.** Human-readable, **pickle-free**
  (safe to share and inspect), and version-tolerant — on a scikit-learn mismatch it *warns*
  instead of failing, and stores the producing library versions. Best for **sharing,
  deploying, archiving, or auditing** a model. Floats round-trip at full double precision.

[OpenModels](https://github.com/Gnpd/openmodels) is purpose-built for this stack: its
documented flagship case is a fitted `chemotools` preprocessing + `PLSRegression` pipeline.
Depth and the security notes are in `references/formats.md`.

## Save and reload (importable)

`save_model` / `load_model` dispatch on the file extension — same call, either format:

```python
from persist_model import save_model, load_model, build_metadata

meta = build_metadata(pipe, n_components=6, x_axis=x_axis, notes="glucose NIR")
save_model(pipe, "pls.joblib")                       # exact, local
save_model(pipe, "pls.json", metadata=meta)          # portable + provenance sidecar
model = load_model("pls.json")                        # format inferred from .json
```

Passing `metadata=` also writes `<path>.meta.json` recording sklearn/chemotools/openmodels
versions and the pipeline steps — provenance that travels with the artifact.

## Convert, verify, inspect (CLI)

```
# turn a local joblib model into a portable JSON one (or the reverse)
python scripts/persist_model.py convert --in pls.joblib --out pls.json

# reload and confirm it still predicts on held-out spectra
python scripts/persist_model.py check --in pls.json --dataset fermentation

# read a JSON model's class/params/attributes WITHOUT reconstructing it
python scripts/persist_model.py inspect --in pls.json
```

The producer scripts speak both formats too: `chemometric-modeling`'s
`fit_model.py --out pls.json --format json` saves JSON directly, and the consumers
`model-diagnostics/applicability_domain.py --model` and `chemometric-validation/metrics.py
--model` load either `.joblib` or `.json`.

## Safety & order discipline

- **Never load an untrusted `.joblib`/`.pkl`** — unpickling runs arbitrary code. JSON is the
  safe interchange format; still, only deserialize JSON from a source you trust.
- **JSON round-trips `.predict()`; transform-based diagnostics need joblib.** A PLS model
  restored from JSON predicts correctly (deployment, `metrics.py`) but currently can't
  `transform` into score space, so the applicability domain / inspector want an exact `joblib`
  model saved at fit time (`fit_model.py --out model.joblib`) — the dropped state can't be
  recovered from the JSON. Details in `references/formats.md`.
- Save the pipeline **after** it is fit on train and **after** you have chosen latent variables
  by CV — persist the model you actually validated, not a re-fit variant.
- The applicability domain is part of a trustworthy deployment: persist the model, then guard
  every prediction with **model-diagnostics** before believing it.

## Scripts

| script | purpose |
|---|---|
| `scripts/persist_model.py` | Save/load a fitted pipeline as joblib or OpenModels JSON; `convert` between them; `check` a reloaded model predicts; `inspect` a JSON model; `build_metadata` provenance sidecar. |

Format trade-offs, security, and version-compatibility depth in `references/formats.md`; the
decision tree in `assets/persistence_decision.svg`.
