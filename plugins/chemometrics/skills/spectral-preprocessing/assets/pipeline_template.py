"""Copy-paste starter: a leakage-free chemotools preprocessing pipeline.

Edit the steps to match your data (see references/technique_catalog.md and
references/ordering_guide.md), then fit on train and transform test.
"""

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.baseline import AirPls
from chemotools.derivative import SavitzkyGolay
from chemotools.feature_selection import RangeCut
from chemotools.scatter import StandardNormalVariate

# --- your data: shape (n_samples, n_features); reshape a single spectrum ---
# X_train, y_train = ...            # training spectra (+ reference values)
# X_test = ...                       # spectra to transform later
# x_axis = ...                       # wavenumbers/wavelengths, shape (n_features,)

# Example with the bundled dataset (requires: pip install pandas):
from chemotools.datasets import load_fermentation_train  # noqa: E402

X_df, y_df = load_fermentation_train()
X_train = X_df.to_numpy(dtype=float)
x_axis = np.asarray(X_df.columns, dtype=float)

# --- build the pipeline (order matters: see ordering_guide.md) ---
pipe = Pipeline(
    [
        ("range_cut", RangeCut(start=950, end=1550, x_axis=x_axis)),
        ("baseline", AirPls(lam=1e5)),
        ("scatter", StandardNormalVariate()),
        ("deriv", SavitzkyGolay(window_length=21, polyorder=2, deriv=1)),
        ("center", StandardScaler(with_mean=True, with_std=False)),
    ]
)

# --- fit on TRAIN only, then apply to any new spectra (no leakage) ---
pipe.fit(X_train)
X_train_p = pipe.transform(X_train)
# X_test_p = pipe.transform(X_test)

print(f"{X_train.shape} -> {X_train_p.shape}")

# --- persist the fitted pipeline for deployment ---
joblib.dump(pipe, "pipeline.joblib")
# later: pipe = joblib.load("pipeline.joblib")
