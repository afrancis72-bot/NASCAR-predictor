# NASCAR Predictor V2.6.1 — Coverage Merge Hotfix

Fixes the V2.6 KeyError on `data_coverage_pct`.

Cause:
The Reference coverage fields were already present upstream in `priors`, so `build_features()` could carry them into `features`. Merging the authoritative coverage frame again created pandas suffix columns (`data_coverage_pct_x` / `_y`), leaving no plain `data_coverage_pct` column.

Fix:
Drop all pre-existing coverage/confidence columns from `features` before merging the authoritative NASCAR Reference values.
No model weights or parser logic changed from V2.6.
