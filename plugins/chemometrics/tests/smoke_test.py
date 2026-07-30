#!/usr/bin/env python
"""Smoke tests for every skill script.

Exercises the core function of each script against small synthetic spectra so it
runs anywhere chemotools + numpy are installed (no pandas/dataset needed). Run
with pytest (``pytest plugins/chemometrics/tests``) or standalone (``python smoke_test.py``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

SKILLS = Path(__file__).resolve().parents[1] / "skills"


def _load(rel: str):
    """Import a script module by path (scripts aren't a package)."""
    path = SKILLS / rel
    name = "skill_" + rel.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(n_samples: int = 8, n_features: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_features)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for c in (0.3, 0.6):
            X[i] += rng.uniform(0.5, 1.5) * np.exp(-0.5 * ((x - c) / 0.03) ** 2)
        X[i] += 0.2 * x + rng.normal(0, 0.01, n_features)  # baseline + noise
    x_axis = np.linspace(1000.0, 1600.0, n_features)
    return X, x_axis


def test_hub_build_pipeline():
    bp = _load("spectral-preprocessing/scripts/build_pipeline.py")
    X, x_axis = _synthetic()
    spec = [
        {"type": "airpls", "params": {"lam": 1e5}},
        {"type": "snv"},
        {"type": "savgol_deriv", "params": {"window_length": 15, "polyorder": 2, "deriv": 1}},
        {"type": "standard_scaler", "params": {"with_mean": True, "with_std": False}},
    ]
    pipe = bp.build_pipeline(spec, x_axis=x_axis)
    out = pipe.fit_transform(X)
    assert out.shape == X.shape
    assert np.isfinite(out).all()


def test_hub_build_pipeline_range_cut_changes_features():
    bp = _load("spectral-preprocessing/scripts/build_pipeline.py")
    X, x_axis = _synthetic()
    spec = [{"type": "range_cut", "params": {"start": 1100, "end": 1500, "x_axis": "$x_axis"}}]
    pipe = bp.build_pipeline(spec, x_axis=x_axis)
    out = pipe.fit_transform(X)
    assert out.shape[1] < X.shape[1]


def test_hub_load_spectra_npy(tmp_path=None):
    ls = _load("spectral-preprocessing/scripts/load_spectra.py")
    X, _ = _synthetic()
    out = Path(tmp_path) if tmp_path else Path(__file__).resolve().parent
    p = out / "_smoke_spectra.npy"
    np.save(p, X)
    try:
        loaded, axis = ls.load_spectra(str(p))
        assert loaded.shape == X.shape
        assert ls.report(loaded, axis)  # non-empty report
    finally:
        os.remove(p)


def test_baseline_tune():
    tb = _load("baseline-scatter-correction/scripts/tune_baseline.py")
    X, _ = _synthetic()
    r = tb.evaluate(X, "airpls", 1e5)
    assert set(r) >= {"lam", "baseline_roughness", "neg_fraction", "residual_offset"}
    assert np.isfinite(r["baseline_roughness"])


def test_baseline_compare_options():
    cc = _load("baseline-scatter-correction/scripts/compare_corrections.py")
    X, _ = _synthetic()
    opts = cc._options()
    for name, t in opts.items():
        Xt = X if t is None else t.fit_transform(X)
        b, r = cc._metrics(Xt)
        assert np.isfinite(b) and np.isfinite(r), name


def test_smoothing_noise_proxy_and_filter():
    ts = _load("smoothing-derivatives/scripts/tune_smoothing.py")
    from chemotools.smooth import SavitzkyGolayFilter

    X, _ = _synthetic()
    raw = ts._noise_proxy(X)
    smoothed = SavitzkyGolayFilter(window_length=15, polyorder=2).fit_transform(X)
    assert ts._noise_proxy(smoothed) <= raw  # smoothing reduces noise proxy


def test_derivative_preview_orders():
    from chemotools.derivative import SavitzkyGolay

    X, _ = _synthetic()
    for deriv in (0, 1, 2):
        out = SavitzkyGolay(window_length=15, polyorder=2, deriv=deriv).fit_transform(X)
        assert out.shape == X.shape
        assert np.isfinite(out).all()


def test_transfer_demo_reduces_gap():
    td = _load("calibration-transfer/scripts/transfer_demo.py")
    from chemotools.adaptation import DirectStandardization

    rng = np.random.default_rng(1)
    src_std, tgt_std = td._make_instrument_pair(rng, 20, 150)
    src_val, tgt_val = td._make_instrument_pair(rng, 30, 150)
    before = td._rmse_between(tgt_val, src_val)
    ds = DirectStandardization().fit(tgt_std, X_source=src_std)
    after = td._rmse_between(ds.transform(tgt_val), src_val)
    assert after < before  # transfer shrinks the instrument gap


def test_align_axes_interpolator():
    import sklearn
    from chemotools.adaptation import XAxisInterpolator

    sklearn.set_config(enable_metadata_routing=True)
    X, x_axis = _synthetic(n_features=200)
    x_common = np.linspace(1050, 1550, 300)
    interp = (
        XAxisInterpolator(common_x_axis=x_common, method="linear", left=0, right=0)
        .set_fit_request(x_axis=True)
        .set_transform_request(x_axis=True)
    )
    out = interp.fit_transform(X, x_axis=x_axis)
    assert out.shape == (X.shape[0], 300)
    assert np.isfinite(out).all()


def test_augment_expands_and_is_reproducible():
    ag = _load("spectral-augmentation/scripts/augment.py")
    X, _ = _synthetic()
    spec = [
        {"type": "add_noise", "params": {"distribution": "gaussian", "scale": 0.01}},
        {"type": "index_shift", "params": {"shift": 3}},
        {"type": "spectrum_scale", "params": {"scale": 0.05}},
    ]
    out1 = ag.augment(X, spec, copies=4, seed=0)
    out2 = ag.augment(X, spec, copies=4, seed=0)
    assert out1.shape == (X.shape[0] * 5, X.shape[1])
    assert np.isfinite(out1).all()
    assert np.allclose(out1, out2)  # same seed -> reproducible
    # original rows preserved as the first block
    assert np.allclose(out1[: X.shape[0]], X)


def test_augment_preview_options():
    pv = _load("spectral-augmentation/scripts/preview_augmentations.py")
    X, _ = _synthetic()
    opts = pv._options(scale=0.02, shift=3, sigma=1.5, seed=0)
    assert set(opts) == {
        "add_noise", "baseline_shift", "spectrum_scale",
        "gaussian_broadening", "index_shift", "fractional_shift",
    }
    for name, t in opts.items():
        out = t.fit_transform(X)
        assert out.shape == X.shape, name
        assert np.isfinite(out).all(), name


def test_modeling_fit_regression_and_classification():
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, _ = _synthetic(n_samples=12)
    y_reg = X[:, 60].copy()  # a band correlated target
    reg = fm.build_full_pipeline("regression", n_components=3)
    reg.fit(X, y_reg)
    assert reg.predict(X).shape[0] == X.shape[0]

    y_cls = np.array(["a", "b"] * 6)
    clf = fm.build_full_pipeline("classification", n_components=2)
    clf.fit(X, y_cls)
    preds = clf.predict(X)
    assert set(np.unique(preds)) <= {"a", "b"}
    assert clf.named_steps["plsda"].predict_proba(X).shape == (X.shape[0], 2)


def test_modeling_fit_with_preprocessing_head():
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, x_axis = _synthetic(n_samples=12)
    y = X[:, 60].copy()
    spec = [{"type": "snv"}, {"type": "savgol_deriv",
            "params": {"window_length": 11, "polyorder": 2, "deriv": 1}}]
    pipe = fm.build_full_pipeline("regression", n_components=3, prep_spec=spec, x_axis=x_axis)
    pipe.fit(X, y)
    assert [n for n, _ in pipe.steps] == ["snv", "savgol_deriv", "pls"]
    assert np.isfinite(pipe.predict(X)).all()


def test_select_components_returns_valid_optimum():
    sc = _load("chemometric-modeling/scripts/select_components.py")
    X, _ = _synthetic(n_samples=20)
    y = X[:, 60] + X[:, 120]
    rows = sc.sweep("regression", X, y, max_components=6, cv=4)
    choice = sc.select(rows)
    n_max = min(6, X.shape[1], X.shape[0] - 1)
    assert 1 <= choice["one_se"] <= choice["min"] <= n_max
    assert all(np.isfinite(r["metric"]) for r in rows)


def test_feature_importance_vip_sr():
    fi = _load("chemometric-modeling/scripts/feature_importance.py")
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, x_axis = _synthetic(n_samples=16)
    y = X[:, 60] + 0.5 * X[:, 120]
    model = fm.build_model("regression", n_components=4)
    model.fit(X, y)
    for method in ("vip", "sr"):
        scores = fi.importance(model, X, y, method=method)
        assert scores.shape == (X.shape[1],)
        assert np.isfinite(scores).all()
    idx, pos = fi.top_bands(fi.importance(model, X, y, method="vip"), x_axis, top=5)
    assert len(idx) == 5 and pos is not None


def test_metrics_regression_and_classification():
    m = _load("chemometric-validation/scripts/metrics.py")
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, 40)
    reg = m.regression_metrics(y, y + rng.normal(0, 0.5, 40))
    assert set(reg) >= {"rmsep", "r2", "rpd", "bias", "sep"}
    assert np.isfinite(reg["rpd"]) and reg["rpd"] > 1  # good fit -> RPD > 1
    yc = np.array(["a", "b", "c"] * 10)
    cls = m.classification_metrics(yc, yc)
    assert cls["accuracy"] == 1.0
    assert np.array(cls["confusion_matrix"]).shape == (3, 3)


def test_cross_validate_runs_leakfree():
    cv = _load("chemometric-validation/scripts/cross_validate.py")
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, _ = _synthetic(n_samples=20)
    y = X[:, 60] + X[:, 120]
    pipe = fm.build_full_pipeline("regression", n_components=3)
    res = cv.run_cv(pipe, X, y, task="regression", cv=4)
    assert set(res) == {"rmse", "r2", "rpd", "bias"}
    assert all(np.isfinite(res[k]["mean"]) for k in res)


def test_grouped_cv_no_group_straddles_fold():
    """Leakage guard: with replicate groups, no group appears in both sides of a split."""
    cv = _load("chemometric-validation/scripts/cross_validate.py")
    X, _ = _synthetic(n_samples=24)
    groups = np.repeat(np.arange(8), 3)  # 8 samples x 3 replicates
    splitter = cv.make_splitter("regression", 4, groups)
    for train_idx, test_idx in splitter.split(X, groups=groups):
        assert not (set(groups[train_idx]) & set(groups[test_idx]))  # disjoint groups


def test_permutation_test_pvalue_in_unit_interval():
    pt = _load("chemometric-validation/scripts/permutation_test.py")
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, _ = _synthetic(n_samples=18)
    y = X[:, 60] + X[:, 120]
    pipe = fm.build_full_pipeline("regression", n_components=3)
    res = pt.permutation_test(pipe, X, y, task="regression", cv=3, n_permutations=20)
    assert 0.0 <= res["p_value"] <= 1.0
    assert len(res["permutation_scores"]) == 20


def test_applicability_domain_flags_out_of_domain():
    ad = _load("model-diagnostics/scripts/applicability_domain.py")
    X, _ = _synthetic(n_samples=16, seed=3)
    model = ad.build_ad_model(X, n_components=4)
    detectors = ad.fit_domain(model, X, confidence=0.95)
    # a deliberately out-of-domain spectrum: rescaled + shifted well beyond train
    X_ood = X[:1] * 5 + 10
    result = ad.flag(detectors, X_ood)
    assert bool(result["outside_any"][0])  # must be flagged
    # in-domain training rows should overwhelmingly pass
    train_flags = ad.flag(detectors, X)
    assert train_flags["outside_any"].mean() < 0.5


def test_diagnose_model_builds_regression_and_pca():
    dm = _load("model-diagnostics/scripts/diagnose_model.py")
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, x_axis = _synthetic(n_samples=14)
    y = X[:, 60] + X[:, 120]
    pls = fm.build_model("regression", n_components=4)
    pls.fit(X, y)
    insp = dm.build_inspector("regression", pls, X, y, x_axis)
    summary = dm.text_summary(insp)
    assert "Hotelling T2 limit" in summary and "n_components" in summary

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3).fit(X)
    insp_pca = dm.build_inspector("pca", pca, X, None, x_axis)
    assert "Q-residuals limit" in dm.text_summary(insp_pca)


def test_scaffold_renders_valid_runnable_scripts():
    sa = _load("chemometrics-workflow/scripts/scaffold_analysis.py")
    # bundled-dataset regression + own-file classification both compile as Python
    reg = sa.render("regression", dataset="fermentation")
    cls = sa.render("classification", data="X.csv", y="y.csv")
    for code in (reg, cls):
        compile(code, "<scaffold>", "exec")  # valid Python
        assert "train_test_split" in code  # split first
        assert "Pipeline(" in code  # one pipeline
        assert "cross_val_score" in code  # LV selection by CV
        assert "HotellingT2" in code and "QResiduals" in code  # AD gate
    assert "_PLSDA" in cls  # classification recipe embedded, no plugin dependency
    assert "StratifiedKFold" in cls and "KFold" in reg


def test_persistence_roundtrip(tmp_path=None):
    """Save/load a fitted pipeline: joblib exact; OpenModels JSON if installed (else skip)."""
    pm = _load("model-persistence/scripts/persist_model.py")
    fm = _load("chemometric-modeling/scripts/fit_model.py")
    X, _ = _synthetic(n_samples=12)
    y = X[:, 60].copy()
    pipe = fm.build_full_pipeline("regression", n_components=3)
    pipe.fit(X, y)
    expected = pipe.predict(X)

    out = Path(tmp_path) if tmp_path else Path(__file__).resolve().parent
    jl = out / "_smoke_model.joblib"
    js = out / "_smoke_model.json"
    try:
        # joblib: exact round-trip (skip only if joblib itself is absent)
        try:
            import joblib  # noqa: F401
        except ImportError:
            print("skip joblib round-trip (joblib not installed)")
        else:
            meta = pm.build_metadata(pipe, n_components=3)
            assert pm.save_model(pipe, str(jl), metadata=meta) == "joblib"
            assert (out / "_smoke_model.joblib.meta.json").exists()  # provenance sidecar
            reloaded = pm.load_model(str(jl))
            assert np.array_equal(reloaded.predict(X), expected)

        # JSON via OpenModels: guarded — keep the suite green when it's not installed
        try:
            import openmodels  # noqa: F401
        except ImportError:
            print("skip JSON round-trip (openmodels not installed)")
        else:
            assert pm.infer_format(str(js)) == "json"
            pm.save_model(pipe, str(js))
            restored = pm.load_model(str(js))
            assert np.allclose(restored.predict(X), expected)
    finally:
        for p in (jl, js, out / "_smoke_model.joblib.meta.json"):
            if p.exists():
                os.remove(p)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
