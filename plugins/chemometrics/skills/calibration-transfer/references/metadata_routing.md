# Metadata routing (dynamic transformers)

Most preprocessing is *static*: the correction is learned at `fit` and the data
alone suffices at `transform`. Some corrections need extra per-call information
that is only known at transform time — a per-spectrum x-axis, a freshly measured
background, a per-measurement scale factor. chemotools delivers this through
scikit-learn's **metadata routing**.

## The pattern

```python
import sklearn
from chemotools.adaptation import XAxisInterpolator

sklearn.set_config(enable_metadata_routing=True)   # opt in (global)

interp = (
    XAxisInterpolator(common_x_axis=x_common, method="linear", left=0, right=0)
    .set_fit_request(x_axis=True)        # declare x_axis is wanted at fit
    .set_transform_request(x_axis=True)  # ... and at transform
)

X_aligned = interp.fit_transform(X, x_axis=per_sample_axes)
```

- `set_config(enable_metadata_routing=True)` turns routing on for the process.
- `set_fit_request(x_axis=True)` / `set_transform_request(x_axis=True)` register
  `x_axis` as a metadata argument for each phase. `fit_transform` goes through
  **both** phases, so declare both.
- The metadata (`x_axis=...`) is then accepted by `fit`/`transform`/`fit_transform`.

## Inside a Pipeline

Routing shines in pipelines: the metadata is delivered **only** to the step that
declared it; every other step is untouched.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from chemotools.scatter import MultiplicativeScatterCorrection

pipe = Pipeline([
    ("interpolate",
     XAxisInterpolator(common_x_axis=x_common, method="linear", left=0, right=0)
       .set_fit_request(x_axis=True)
       .set_transform_request(x_axis=True)),
    ("msc", MultiplicativeScatterCorrection()),
    ("scaler", StandardScaler()),
])

# x_axis reaches only "interpolate"; msc and scaler never see it
X_out = pipe.fit_transform(X, x_axis=per_sample_axes)
```

## Shared vs. per-sample grids

`x_axis` accepts two shapes:

- `(n_features,)` — one grid shared by every spectrum in the call (single
  instrument / session).
- `(n_samples, n_features)` — one grid per spectrum (combining instruments in one
  batch).

## Why this matters

Without routing you would have to bake the axis into the transformer at `fit`,
which breaks the moment a new spectrum arrives on a different grid. Routing keeps
the pipeline **self-contained and reusable** even when correction parameters are
not known until prediction time. The same mechanism will back future dynamic
transformers (e.g. per-sample background subtraction, normalize-by-laser-power).
