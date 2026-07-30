---
description: Run an end-to-end chemometrics analysis of a spectral dataset (explore → preprocess → model → validate → deploy).
argument-hint: "[dataset name or path to spectra] [regression|classification]"
---

Run a complete chemometrics analysis by following the **chemometrics-workflow**
skill as the single source of truth for the procedure. Load that skill and apply
its five-stage sequence and leakage discipline.

Target: **$ARGUMENTS**
(A bundled dataset name — `fermentation` or `coffee` — or a path to a spectra
file, optionally followed by the task. If the task is not given, infer it from
the target/target column and confirm with the user.)

Steps:

1. **Scope & context.** Confirm this is an end-to-end request. Establish
   modality, data shape / x-axis, the task (regression / classification /
   exploration), and whether replicate or augmented spectra exist (→ grouped CV).
   **Split a held-out test set first** — everything downstream is fit on train only.
2. **Scaffold.** Generate a runnable starter with
   `skills/chemometrics-workflow/scripts/scaffold_analysis.py` for the resolved
   dataset + task. Use it as the correct-by-construction skeleton.
3. **Deepen each stage** by routing to the owning spoke (do not re-derive their
   logic): preprocessing → `spectral-preprocessing`; model + latent-variable
   selection → `chemometric-modeling`; leakage-free CV, metrics, permutation test
   → `chemometric-validation`; applicability domain → `model-diagnostics`.
4. **Report.** Present the chosen pipeline, the CV and held-out metrics
   (RMSEP/R²/RPD/bias or accuracy/F1/confusion), the permutation-test result when
   `n` is small vs. the number of bands, and the applicability-domain summary.
   State every assumption and each place the user should confirm a choice.

Runtime: the scripts need `chemotools[viz]` (+ `pandas` for file I/O). If imports
fail, tell the user to `pip install "chemotools[viz]" pandas` and stop.
