# Metrics cheat-sheet

## Regression — report all of these
```
RMSEP = sqrt(mean((y_pred - y_true)^2))      # units of y; lower is better
R^2   = 1 - SS_res / SS_tot                  # variance explained
RPD   = std(y_true) / RMSEP                  # >2 useful, >3 good, >5 excellent
bias  = mean(y_pred - y_true)                # ~0 wanted; persistent bias = drift
SEP   = std(y_pred - y_true, ddof=1)         # bias-corrected: RMSEP^2 ~ bias^2 + SEP^2
```

## Classification
```
accuracy      = correct / n                  # misleading if imbalanced
macro-F1      = mean of per-class F1          # prefer under imbalance
confusion[i,j]= true class i predicted as j  # always read it
```

## Which fold splitter
```
replicates / augmented copies present  -> GroupKFold(groups=...)   # mandatory
classification                         -> StratifiedKFold
plain regression, independent samples  -> KFold(shuffle=True)
```

## Sanity checks
```
RMSEP >> RMSECV        -> over-fit to CV, or test-set distribution shift (check AD)
grouped CV << ungrouped-> the ungrouped score was leaking; grouped is honest
high R^2 + high bias   -> broken for quantitation despite "looking" accurate
high score, p >= 0.05  -> chance correlation; not significant
```

## One-liners (this skill)
```
python scripts/cross_validate.py  --task regression     --dataset fermentation --n-components 6 --groups g.npy
python scripts/permutation_test.py --task classification --dataset coffee       --n-components 4 --n-permutations 200
python scripts/metrics.py          --task regression     --dataset fermentation --model pls.joblib
```
