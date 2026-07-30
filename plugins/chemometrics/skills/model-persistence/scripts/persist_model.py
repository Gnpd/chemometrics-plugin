#!/usr/bin/env python
"""Persist a fitted chemometric pipeline — joblib (exact) or OpenModels JSON (portable).

Two formats, chosen by file extension:

* ``.joblib`` / ``.pkl`` — :mod:`joblib` pickle. **Exact** bit-for-bit round-trip,
  fast, the default. Binary and pickle-based: only load models you trust, and it
  can break across scikit-learn / chemotools versions.
* ``.json``             — `OpenModels <https://github.com/Gnpd/openmodels>`_
  serialization. Human-readable, **pickle-free** (safe to share), version-tolerant
  (warns on a scikit-learn mismatch instead of breaking), and records the producing
  library versions. Floats round-trip to full double precision. Needs
  ``openmodels`` installed; to round-trip chemotools transformers it also uses
  ``chemotools.utils.discovery.all_estimators`` (chemotools >= 0.2.2).

Use joblib for local reuse in the same environment; export JSON to share, deploy,
archive, or inspect a model.

Importable (the smoke test calls :func:`save_model` / :func:`load_model`) and
CLI-runnable::

    # convert an existing joblib model to portable JSON (and back)
    python persist_model.py convert --in pls.joblib --out pls.json

    # reload a saved model and verify it predicts on held-out spectra
    python persist_model.py check --in pls.json --dataset fermentation

    # peek at a serialized model without fully trusting/executing it
    python persist_model.py inspect --in pls.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

_JOBLIB_EXT = {".joblib", ".pkl"}
_JSON_EXT = {".json"}


def infer_format(path: str | Path, fmt: str = "auto") -> str:
    """Resolve a format name (``"joblib"`` | ``"json"``) from *fmt* or the extension."""
    if fmt and fmt != "auto":
        if fmt not in ("joblib", "json"):
            raise ValueError(f"Unknown format {fmt!r}; use joblib | json | auto.")
        return fmt
    ext = Path(path).suffix.lower()
    if ext in _JSON_EXT:
        return "json"
    if ext in _JOBLIB_EXT:
        return "joblib"
    # default: joblib (the exact, dependency-light format)
    return "joblib"


def _json_manager():
    """Build an OpenModels manager that also understands chemotools estimators.

    Guards both optional imports with actionable messages. Without chemotools's
    ``all_estimators`` a pipeline containing chemotools transformers cannot be
    round-tripped, so we warn rather than fail silently.
    """
    try:
        from openmodels import SerializationManager, SklearnSerializer
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        raise SystemExit(
            "JSON persistence needs OpenModels: pip install openmodels "
            "(it is pre-release; use `pip install --pre openmodels` if pip finds no version)."
        ) from exc

    custom_estimators = None
    try:
        from chemotools.utils.discovery import all_estimators as _chemo_all

        custom_estimators = _chemo_all
    except Exception:  # noqa: BLE001 - any failure means no chemotools discovery
        warnings.warn(
            "chemotools.utils.discovery.all_estimators unavailable (needs chemotools >= 0.2.2); "
            "JSON round-trip will only cover plain scikit-learn steps, not chemotools transformers.",
            stacklevel=2,
        )
    return SerializationManager(SklearnSerializer(custom_estimators=custom_estimators))


def build_metadata(
    model: Any,
    *,
    n_components: int | None = None,
    x_axis: Any = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Provenance to store next to a saved model (``<path>.meta.json``).

    Records the producing library versions and the pipeline shape so a reloaded
    model can be traced back to how it was fit. Mirrors OpenModels' own
    ``producer_version`` at the plugin level.
    """
    meta: dict[str, Any] = {"versions": {}}
    for lib in ("sklearn", "chemotools", "openmodels", "numpy", "scipy"):
        try:
            meta["versions"][lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001 - optional / absent libraries
            pass
    steps = getattr(model, "steps", None)
    if steps is not None:
        meta["pipeline_steps"] = [name for name, _ in steps]
    meta["estimator_class"] = type(model).__name__
    if n_components is not None:
        meta["n_components"] = int(n_components)
    if x_axis is not None:
        try:
            meta["n_features"] = int(len(x_axis))
        except TypeError:
            pass
    if notes:
        meta["notes"] = notes
    return meta


def save_model(
    model: Any,
    path: str | Path,
    fmt: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Persist *model* to *path*; returns the resolved format name.

    ``fmt="auto"`` (default) picks joblib for ``.joblib``/``.pkl`` and OpenModels
    JSON for ``.json``. If *metadata* is given, also writes ``<path>.meta.json``.
    """
    resolved = infer_format(path, fmt)
    path = str(path)
    if resolved == "joblib":
        try:
            import joblib
        except ImportError as exc:
            raise SystemExit("joblib persistence needs joblib: pip install joblib.") from exc
        joblib.dump(model, path)
    else:
        _json_manager().save(model, path, format_name="json")
    if metadata is not None:
        Path(path + ".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return resolved


def load_model(path: str | Path, fmt: str = "auto") -> Any:
    """Load a model saved by :func:`save_model` (format inferred from the extension)."""
    resolved = infer_format(path, fmt)
    path = str(path)
    if resolved == "joblib":
        try:
            import joblib
        except ImportError as exc:
            raise SystemExit("Loading a joblib model needs joblib: pip install joblib.") from exc
        return joblib.load(path)
    return _json_manager().load(path, format_name="json")


def convert(in_path: str | Path, out_path: str | Path) -> str:
    """Load a model from one format and re-save it in the other. Returns the out format."""
    model = load_model(in_path)
    return save_model(model, out_path)


# --------------------------------------------------------------------------- CLI


def _load_x(args) -> Any:
    """Resolve spectra for a round-trip check from --dataset or --x (self-contained)."""
    import numpy as np

    if getattr(args, "dataset", None):
        from chemotools import datasets

        loaders = {
            "fermentation": datasets.load_fermentation_test,
            "coffee": datasets.load_coffee,
        }
        X, _ = loaders[args.dataset]()
        return np.asarray(X, dtype=float)
    ext = Path(args.x).suffix.lower()
    if ext == ".npy":
        return np.load(args.x)
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Reading CSV/Parquet spectra needs pandas: pip install pandas.") from exc
    df = pd.read_parquet(args.x) if ext == ".parquet" else pd.read_csv(args.x)
    return df.to_numpy(dtype=float)


def _cmd_convert(args) -> None:
    out_fmt = convert(args.inp, args.out)
    print(f"converted {args.inp} -> {args.out} ({out_fmt})")


def _cmd_check(args) -> None:
    import numpy as np

    model = load_model(args.inp)
    print(f"loaded {type(model).__name__} from {args.inp}")
    if not (args.dataset or args.x):
        print("(no --dataset/--x given; load-only check passed)")
        return
    X = _load_x(args)
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred)
    finite = bool(np.isfinite(y_pred).all()) if y_pred.dtype.kind in "fc" else True
    print(f"predicted {y_pred.shape} on X{X.shape}; all finite: {finite}")


def _cmd_inspect(args) -> None:
    """Show a model's shape. For JSON, read the payload without reconstructing it."""
    ext = Path(args.inp).suffix.lower()
    if ext in _JSON_EXT:
        data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
        keys = ("estimator_class", "producer_name", "producer_version", "domain")
        for k in keys:
            if k in data:
                print(f"{k}: {data[k]}")
        params = data.get("params", {})
        if isinstance(params, dict):
            print(f"params: {sorted(params)}")
        attrs = data.get("attributes")
        if isinstance(attrs, dict):
            print(f"fitted attributes: {sorted(attrs)[:20]}")
    else:
        print(repr(load_model(args.inp)))
    meta = Path(str(args.inp) + ".meta.json")
    if meta.exists():
        print("--- provenance (.meta.json) ---")
        print(meta.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Load a model and re-save it in the other format.")
    p_conv.add_argument("--in", dest="inp", required=True, help="Source model (.joblib/.pkl/.json).")
    p_conv.add_argument("--out", required=True, help="Destination model (extension picks format).")
    p_conv.set_defaults(func=_cmd_convert)

    p_chk = sub.add_parser("check", help="Reload a model and (optionally) predict to verify it.")
    p_chk.add_argument("--in", dest="inp", required=True, help="Model to reload.")
    p_chk.add_argument("--dataset", choices=["fermentation", "coffee"], help="Bundled spectra to predict on.")
    p_chk.add_argument("--x", help="Spectra file (.npy/.csv/.parquet) to predict on.")
    p_chk.set_defaults(func=_cmd_check)

    p_ins = sub.add_parser("inspect", help="Show a serialized model's class/params/attributes.")
    p_ins.add_argument("--in", dest="inp", required=True, help="Model to inspect.")
    p_ins.set_defaults(func=_cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
