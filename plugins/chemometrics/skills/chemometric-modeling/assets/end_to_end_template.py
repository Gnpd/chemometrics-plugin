#!/usr/bin/env python
"""End-to-end modeling starter: preprocess -> PLS -> select LVs -> fit -> predict.

Copy this into your project and edit the marked spots. It uses only the public
chemotools + scikit-learn API and follows the plugin's leakage discipline:
split first, build ONE Pipeline, choose LVs by CV on train, fit on train,
predict on the held-out test set. Swap the bundled loader for your own data.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

from chemotools.regression import PLSRegression
from chemotools.scatter import StandardNormalVariate
from chemotools.derivative import SavitzkyGolay

# 1. Load data ---------------------------------------------------------------
# EDIT: replace with your spectra (X: n_samples x n_bands) and target y.
from chemotools.datasets import load_fermentation_train

Xdf, ydf = load_fermentation_train()
X, y = Xdf.to_numpy(), ydf.to_numpy().ravel()

# 2. Split FIRST — everything below is fit on train only ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 3. One Pipeline: preprocessing head + PLS model ----------------------------
# EDIT the preprocessing head using the spectral-preprocessing skill's guidance.
def make_pipeline(n_components: int) -> Pipeline:
    return Pipeline([
        ("snv", StandardNormalVariate()),
        ("deriv", SavitzkyGolay(window_length=15, polyorder=2, deriv=1)),
        ("pls", PLSRegression(n_components=n_components)),
    ])

# 4. Choose the number of latent variables by CV (1-SE rule) -----------------
n_splits = 5
cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)  # GroupKFold for replicates
rows = []
# PLS can use at most (smallest fold-train size) LVs; cap the sweep accordingly.
max_lv = min(15, X_train.shape[1], X_train.shape[0] * (n_splits - 1) // n_splits - 1)
for nc in range(1, max_lv + 1):
    scores = cross_val_score(make_pipeline(nc), X_train, y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
    rows.append((nc, -scores.mean(), scores.std(ddof=1) / np.sqrt(len(scores))))

best_nc, best_rmse, best_se = min(rows, key=lambda r: r[1])
one_se = min(nc for nc, m, _ in rows if m <= best_rmse + best_se)  # parsimonious
print(f"best RMSECV {best_rmse:.3f} at {best_nc} LV; 1-SE choice = {one_se} LV")

# 5. Fit the final model on train, evaluate on the held-out test set ---------
model = make_pipeline(one_se).fit(X_train, y_train)
y_pred = model.predict(X_test).ravel()
print(f"RMSEP {root_mean_squared_error(y_test, y_pred):.3f}  R2 {r2_score(y_test, y_pred):.3f}")

# 6. Next steps --------------------------------------------------------------
# - chemometric-validation : permutation test, RPD/bias, grouped CV.
# - model-diagnostics      : applicability domain — is a new spectrum safe to predict?
# - persist: import joblib; joblib.dump(model, "model.joblib")
