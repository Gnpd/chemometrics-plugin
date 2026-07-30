# chemotools / scikit-learn API contract

What every chemotools transformer guarantees, and how to compose them.

## The transformer interface

Every preprocessing class is a scikit-learn transformer:

```python
from chemotools.baseline import AirPls

t = AirPls(lam=1e5)          # __init__ only stores params (no work, no validation)
t.fit(X_train)               # learns any needed state; returns self
X_corr = t.transform(X_new)  # applies it; returns a new array
X_corr = t.fit_transform(X_train)   # fit + transform in one call (training only)
```

- **Input:** `X` is a 2-D float array `(n_samples, n_features)`. Rows are spectra,
  columns are spectral channels. Reshape single spectra with `x.reshape(1, -1)`.
- **`y` is ignored** by unsupervised steps (baseline, scatter, smooth, derivative,
  scale) but accepted for API compatibility. Supervised steps (`OSC`, `EPO`,
  `SRSelector`, `VIPSelector`) use `y`.
- **Output shape:** most steps are one-to-one (`n_features` unchanged).
  `RangeCut`/`IndexSelector` change `n_features`. Plan index-based steps around
  this.
- **`get_params` / `set_params` / `clone`** work because `__init__` stores
  parameters verbatim — so these plug into `GridSearchCV`, `RandomizedSearchCV`.

## Pipelines

Compose steps with a `Pipeline` (or `make_pipeline`); the pipeline *is* the
model and the deployable unit:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from chemotools.baseline import AirPls
from chemotools.scatter import StandardNormalVariate
from chemotools.derivative import SavitzkyGolay

pipe = make_pipeline(
    AirPls(lam=1e5),
    StandardNormalVariate(),
    SavitzkyGolay(window_length=21, polyorder=2, deriv=1),
    StandardScaler(with_mean=True, with_std=False),
)
pipe.fit(X_train)                 # fits every step in order, on training data
X_train_p = pipe.transform(X_train)
X_test_p = pipe.transform(X_test) # same learned parameters applied to test
```

Add a model as the final step (`PLSRegression`, `PCA`, ...) and the whole thing
becomes one estimator you can cross-validate and tune.

`scripts/build_pipeline.py` builds exactly this from a declarative spec so you
don't hand-write imports; `scripts/preprocess.py` wraps fit-on-train /
transform-on-input and persistence.

## Current vs deprecated parameter names

chemotools renamed several parameters; the old names still work but emit a
`FutureWarning`. **Use the new names** (the scripts and catalog already do):

| deprecated | current |
|---|---|
| `wavenumbers=` | `x_axis=` |
| `window_size=` | `window_length=` |
| `polynomial_order=` | `polyorder=` (Savitzky-Golay) |
| `derivate_order=` / `derivative_order=` | `deriv=` |

Older tutorials (including some in the chemotools docs) use the deprecated names.

## Metadata routing (dynamic transformers)

Some transformers need per-call information at `transform` time (e.g. a
per-sample x-axis for `XAxisInterpolator`, a fresh background). scikit-learn
delivers it via metadata routing:

```python
import sklearn
sklearn.set_config(enable_metadata_routing=True)

interp = (XAxisInterpolator(common_x_axis=x_common, method="linear", left=0, right=0)
          .set_fit_request(x_axis=True)
          .set_transform_request(x_axis=True))
X_aligned = interp.fit_transform(X, x_axis=per_sample_axes)
```

Only the step that declared the request receives the metadata; others are
unaffected. Full treatment in the **calibration-transfer** skill.

## Persistence

Persist the fitted pipeline with `joblib.dump(pipe, "pipe.joblib")` and reload
with `joblib.load`. chemotools transformers implement `__setstate__` for
backward-compatible unpickling across the parameter renames above.

For a portable, pickle-free alternative — human-readable JSON that is safe to share
and version-tolerant — export via the **model-persistence** skill
(`save_model(pipe, "pipe.json")`, backed by OpenModels). See that skill's
`references/formats.md` for the joblib-vs-JSON trade-off and security notes.

## Reproducible sample data

chemotools ships datasets for examples (require `pip install pandas`):

```python
from chemotools.datasets import load_fermentation_train, load_fermentation_test, load_coffee
X_train, y_train = load_fermentation_train()   # 21 ATR-FTIR spectra + glucose (HPLC)
X_test,  y_test  = load_fermentation_test()     # >1000 on-line spectra
spectra, labels  = load_coffee()                # ATR-FTIR, classification
```

Wavenumbers are the DataFrame column names (floats). `set_output="polars"` is
also supported.
