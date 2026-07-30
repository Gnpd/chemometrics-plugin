# Persistence formats: joblib vs OpenModels JSON

Two ways to persist a fitted `Pipeline`. They are complementary — joblib for exact local
reuse, JSON for anything that leaves the machine.

## At a glance

| | `joblib` (`.joblib`) | OpenModels JSON (`.json`) |
|---|---|---|
| Fidelity | Exact, bit-for-bit | Approximate in theory; float64 round-trips at full `repr` precision in practice |
| Readable / inspectable | No (binary pickle) | Yes (plain JSON: class, params, fitted attributes) |
| Safety on load | Pickle → **arbitrary code execution** risk | Pickle-free; reconstructs known estimator classes and sets attributes |
| Cross-version | Fragile (breaks on sklearn/chemotools drift) | Tolerant — **warns** on a scikit-learn mismatch, does not break; stores `producer_version` |
| chemotools transformers | Works (chemotools ships `__setstate__` for back-compat unpickling) | Needs `custom_estimators=all_estimators` (chemotools ≥ 0.2.2) |
| Dependency | `joblib` | `openmodels` (+ scikit-learn) |
| Best for | Local reuse, same environment, exact reproduction | Sharing, deploying, archiving, auditing, cross-environment |

**Rule of thumb:** default to joblib; export JSON the moment the model leaves your machine.

## How the JSON path works

The plugin builds an OpenModels manager that also understands chemotools estimators:

```python
from openmodels import SerializationManager, SklearnSerializer
from chemotools.utils.discovery import all_estimators   # chemotools >= 0.2.2

manager = SerializationManager(SklearnSerializer(custom_estimators=all_estimators))
manager.save(pipe, "pls.json", format_name="json")       # serialize + write
model = manager.load("pls.json", format_name="json")      # read + reconstruct
```

`persist_model.py` wraps exactly this (`_json_manager()`), guarding both imports:

- **No `openmodels`** → actionable error (`pip install openmodels`; it is pre-release, so
  `pip install --pre openmodels` if pip resolves no version).
- **No chemotools `all_estimators`** (older chemotools) → a warning, and the JSON round-trip
  then covers only plain scikit-learn steps, not chemotools transformers.

The same `custom_estimators` mapping is needed at **load** time to rebuild chemotools steps, so
always reload chemotools+sklearn JSON through `persist_model.load_model` (or the wrapped
manager), not a bare OpenModels manager.

### What the JSON contains

Per estimator (recursively for every pipeline step): `estimator_class`, constructor `params`
(+ `param_types`/`param_dtypes`), `producer_name`/`producer_version` (`sklearn.__version__` at
save time), `domain`, and — once fitted — `attributes` with the fitted state (`coef_`,
`intercept_`, `classes_`, PLS's `_x_mean`, …), numpy arrays encoded as nested JSON lists with
their dtype recorded separately. A `Pipeline` serializes as a `BaseEstimator` whose `steps`
param holds the nested step estimators.

## Round-trip coverage and edge cases

- **Well supported:** a fitted `Pipeline` of chemotools preprocessing + `PLSRegression` / `PCA`
  / a scikit-learn classifier — the plugin's normal output. Broadly tested across the
  scikit-learn estimator zoo; tested against scikit-learn **1.6.1 / 1.7.2 / 1.8.0**.
- **Not supported by OpenModels:** `PatchExtractor`, `LocalOutlierFactor`; custom (unregistered)
  score functions; kernels/losses outside its registries. Deserializing an unknown class raises
  `UnsupportedEstimatorError`.
- **Version mismatch** only *warns* — verify predictions after loading across environments.
- Plain integer arrays without a stored dtype can default to `int32`; not a concern for the
  float spectra/coefficients in chemometric models.

### `.predict()` vs `.transform()` — a JSON boundary to know

With OpenModels `0.1.0a21`, a `PLSRegression` restored from JSON **predicts** correctly but
**cannot `transform`**: the serializer keeps `_x_mean` but not `_x_std` / `_y_mean` / `_y_std`,
which `transform` needs (`AttributeError: '... ' object has no attribute '_x_std'`). Practical
consequence for this plugin:

- **Predict-based use round-trips through JSON** — deployment scoring and
  `chemometric-validation/metrics.py --model model.json` work.
- **Transform-based diagnostics need joblib saved at fit time** —
  `model-diagnostics/applicability_domain.py` projects spectra into score space, so a JSON PLS
  model there raises a clear error. Note the dropped scaling state **cannot be recovered by
  converting the JSON back to joblib** (the JSON never stored it); persist the model as joblib
  directly from the fit (`fit_model.py --out model.joblib`) when you need diagnostics.

joblib has no such gap (exact). This is an OpenModels serialization bug, not a plugin one — the
fix is to add `_x_std`, `_y_mean`, `_y_std` to the PLS attribute-exception list (mirroring the
`PLSSVD` entry that already lists `_x_std`); once released, JSON will round-trip `transform` too.

## Security

- **joblib/pickle executes arbitrary code on load.** Never `load_model` a `.joblib`/`.pkl` from
  an untrusted source. Treat model files like executables.
- **JSON is the safe interchange format** — no bytecode, reconstruction is class-whitelisted.
  Still deserialize JSON only from sources you trust: the format can carry serialized *function
  references* that OpenModels re-imports by module + name on load.
- This is OpenModels' reason to exist: a transparent, inspectable, pickle-free alternative to
  shipping pickles.

## Provenance sidecar

`build_metadata(model, n_components=…, x_axis=…, notes=…)` captures sklearn/chemotools/openmodels
(+ numpy/scipy) versions, the pipeline step names, `n_components`, and feature count. Passing it
to `save_model(..., metadata=meta)` writes `<path>.meta.json` next to the model — so a saved
artifact records *how* it was produced, independent of the format's own `producer_version`.
