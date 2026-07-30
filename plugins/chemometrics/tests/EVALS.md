# Scenario evals

Manual routing/behavior checks for the plugin (plan Part 5). Each prompt should
activate the named skill(s) and produce a **leakage-free, applicability-domain–
guarded** result. Run these in a Claude Code session with the plugin installed
(or the skills copied to `~/.claude/skills/`). These are qualitative — there is
no automated harness; the automated guarantees live in `smoke_test.py`,
`check_frontmatter.py`, and `check_manifests.py`.

| # | Prompt | Should route to | Pass criteria |
|---|---|---|---|
| 1 | "Build and validate a PLS model for glucose from these raw spectra." | `chemometrics-workflow` → modeling + validation | Splits first; one Pipeline; LVs chosen by CV (1-SE), not training fit; reports RMSEP/R²/RPD/bias; AD mentioned. |
| 2 | "Classify these coffee spectra by origin." | `chemometrics-workflow` / `chemometric-modeling` | PLS-DA (one-hot + argmax); stratified CV; accuracy/F1/confusion; permutation test given small n. |
| 3 | "Is this new spectrum safe to predict?" | `model-diagnostics` | Fits AD on train; Hotelling T² + Q-residuals; flags out-of-domain; does not silently score. |
| 4 | "How many latent variables should I use?" | `chemometric-modeling` | CV sweep; 1-SE parsimonious choice; warns against picking by training R². |
| 5 | "Cross-validate this model — I have 3 replicates per sample." | `chemometric-validation` | Insists on `GroupKFold`; explains ungrouped CV leaks; metrics as mean ± std. |
| 6 | "Which baseline correction should I use?" | `spectral-preprocessing` / `baseline-scatter-correction` | Defers to the single spoke — does **not** launch the whole workflow. |
| 7 | "Analyze this spectral dataset end to end." | `chemometrics-workflow` | Full five-stage sequence; offers `scaffold_analysis.py`; routes each stage to its owner. |
| 8 | "Is my model actually significant or just fitting noise?" | `chemometric-validation` | Runs the y-scramble permutation test; reports a p-value and interprets it. |

Trigger-hygiene expectation: single-stage prompts (6) hit the owning spoke
directly; whole-dataset prompts (1, 2, 7) hit the orchestrator. The automated
`check_frontmatter.py` guards that spoke descriptions stay non-overlapping.
