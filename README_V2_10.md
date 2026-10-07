# NASCAR Predictor V2.10 — DNA Handoff Fix

V2.10 explicitly hands the scored NASCAR Reference columns from `reference_dna`
to the priors dataframe consumed by `build_features()`.

It also adds observed-data fallbacks inside the engine:
- recent form -> recent average finish
- place differential -> recent place differential
- track fit -> comparable-track average finish
- track Elo -> superspeedway Elo
- reliability -> season average finish

Fallbacks are used only if the scored component has no usable cross-field signal.
No missing value is converted into poor performance.

A hard PASS/FAIL integration audit now reports cross-field standard deviation.
Do not use final simulation output unless the audit passes.
