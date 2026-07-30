#!/usr/bin/env python
"""Build a scikit-learn Pipeline of chemotools preprocessing steps from a spec.

A *spec* is a list of steps. Each step is a mapping with a ``type`` (a short
name from the registry below) and optional ``params``. The spec can be written
in YAML/JSON (see ``load_spec``) or passed as a Python list of dicts.

Example spec (YAML)::

    - type: range_cut
      params: {start: 950, end: 1550, x_axis: $x_axis}
    - type: airpls
      params: {lam: 100000.0}
    - type: snv
    - type: savgol_deriv
      params: {window_length: 21, polyorder: 2, deriv: 1}
    - type: standard_scaler
      params: {with_mean: true, with_std: false}

The sentinel string ``$x_axis`` in any param is replaced at build time with the
x-axis array passed to :func:`build_pipeline` (wavenumbers / wavelengths). This
lets a single spec stay reusable across datasets that share a technique but not
a grid.

Run directly to print the pipeline built from a spec file::

    python build_pipeline.py --spec spec.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Callable

# Registry: short spec name -> (module, class name).
# Only public chemotools/sklearn classes; current (non-deprecated) parameters.
REGISTRY: dict[str, tuple[str, str]] = {
    # physics / units
    "intensity_conversion": ("chemotools.physics", "IntensityConversion"),
    # feature selection / region
    "range_cut": ("chemotools.feature_selection", "RangeCut"),
    "index_selector": ("chemotools.feature_selection", "IndexSelector"),
    # baseline
    "airpls": ("chemotools.baseline", "AirPls"),
    "arpls": ("chemotools.baseline", "ArPls"),
    "asls": ("chemotools.baseline", "AsLs"),
    "polynomial_correction": ("chemotools.baseline", "PolynomialCorrection"),
    "linear_correction": ("chemotools.baseline", "LinearCorrection"),
    "constant_baseline_correction": (
        "chemotools.baseline",
        "ConstantBaselineCorrection",
    ),
    "cubic_spline_correction": ("chemotools.baseline", "CubicSplineCorrection"),
    "rubberband_correction": ("chemotools.baseline", "RubberbandCorrection"),
    "non_negative": ("chemotools.baseline", "NonNegative"),
    "subtract_reference": ("chemotools.baseline", "SubtractReference"),
    # scatter / normalization
    "snv": ("chemotools.scatter", "StandardNormalVariate"),
    "rnv": ("chemotools.scatter", "RobustNormalVariate"),
    "msc": ("chemotools.scatter", "MultiplicativeScatterCorrection"),
    "emsc": ("chemotools.scatter", "ExtendedMultiplicativeScatterCorrection"),
    # smoothing
    "savgol_filter": ("chemotools.smooth", "SavitzkyGolayFilter"),
    "whittaker": ("chemotools.smooth", "WhittakerSmooth"),
    "mean_filter": ("chemotools.smooth", "MeanFilter"),
    "median_filter": ("chemotools.smooth", "MedianFilter"),
    "modified_sinc_filter": ("chemotools.smooth", "ModifiedSincFilter"),
    # derivatives
    "savgol_deriv": ("chemotools.derivative", "SavitzkyGolay"),
    "norris_williams": ("chemotools.derivative", "NorrisWilliams"),
    # scaling
    "min_max_scaler": ("chemotools.scale", "MinMaxScaler"),
    "norm_scaler": ("chemotools.scale", "NormScaler"),
    "pareto_scaler": ("chemotools.scale", "ParetoScaler"),
    "point_scaler": ("chemotools.scale", "PointScaler"),
    "band_scaler": ("chemotools.scale", "BandScaler"),
    # orthogonal signal correction
    "osc": ("chemotools.projection", "OrthogonalSignalCorrection"),
    "direct_orthogonalization": ("chemotools.projection", "DirectOrthogonalization"),
    "epo": ("chemotools.projection", "ExternalParameterOrthogonalization"),
    # axis alignment (see the calibration-transfer skill for full guidance)
    "x_axis_interpolator": ("chemotools.adaptation", "XAxisInterpolator"),
    # sklearn helpers commonly used as the final preprocessing step
    "standard_scaler": ("sklearn.preprocessing", "StandardScaler"),
}


def _resolve_class(step_type: str) -> Callable[..., Any]:
    try:
        module_name, class_name = REGISTRY[step_type]
    except KeyError as exc:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown step type {step_type!r}. Known types: {known}"
        ) from exc
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_params(params: dict[str, Any], x_axis: Any) -> dict[str, Any]:
    """Replace ``$x_axis`` sentinels with the provided x-axis array."""
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value == "$x_axis":
            if x_axis is None:
                raise ValueError(
                    f"Param {key!r} requested $x_axis but no x_axis was provided "
                    "to build_pipeline()."
                )
            resolved[key] = x_axis
        else:
            resolved[key] = value
    return resolved


def build_pipeline(spec: list[dict[str, Any]], x_axis: Any = None):
    """Construct a scikit-learn ``Pipeline`` from *spec*.

    Parameters
    ----------
    spec : list of dict
        Each dict has ``type`` (registry key) and optional ``params``.
    x_axis : array-like or None
        Substituted for any ``$x_axis`` param sentinel.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.pipeline import Pipeline

    steps = []
    seen: dict[str, int] = {}
    for entry in spec:
        step_type = entry["type"]
        params = _resolve_params(entry.get("params", {}) or {}, x_axis)
        cls = _resolve_class(step_type)
        # unique, readable step names (range_cut, range_cut-2, ...)
        seen[step_type] = seen.get(step_type, 0) + 1
        name = step_type if seen[step_type] == 1 else f"{step_type}-{seen[step_type]}"
        steps.append((name, cls(**params)))
    return Pipeline(steps)


def load_spec(path: str) -> list[dict[str, Any]]:
    """Load a spec from a ``.yaml``/``.yml`` or ``.json`` file."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading YAML specs requires PyYAML: pip install pyyaml "
                "(or use a .json spec)."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", required=True, help="Path to a YAML/JSON spec file.")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    # Structure-only print: supply a placeholder x-axis so $x_axis steps resolve.
    needs_axis = any(
        v == "$x_axis" for s in spec for v in (s.get("params") or {}).values()
    )
    x_axis = np.arange(2.0) if needs_axis else None
    if needs_axis:
        print("(note: $x_axis shown as placeholder; real axis is injected at runtime)\n")
    print(build_pipeline(spec, x_axis=x_axis))


if __name__ == "__main__":
    main()
