# NASCAR Predictor V2.11 — Pipeline Trace

This is a diagnostic build. It does not change model weights.

Driver DNA now exposes three checkpoints:
1. NASCAR Reference scored output (`reference_dna`)
2. exact `priors` dataframe immediately before `build_features()`
3. exact Reference audit signals immediately after `build_features()`

An automatic TRACE RESULT identifies whether the break is:
- Reference scoring,
- merge/handoff, or
- `build_features()` normalization/wiring.

Use this build to locate the fault before making another model change.
