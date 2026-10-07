# NASCAR Predictor V2.13 — Cache Refresh Fix

Root cause identified: the live Extraction Diagnostic called the current parser directly,
but the 39-driver batch Driver DNA path was cached for six hours. The cache key did not
include a parser version, so a dataframe created before component extraction worked could
survive later deployments. That stale frame had a DK-driven overall Driver DNA score but
empty Reference components — exactly matching the audit pattern.

V2.13:
- adds parser version 2.13 to the Streamlit cache key, forcing a fresh 39-driver pull;
- validates Checkpoint 1 using only the five NASCAR Reference components, not overall DNA;
- supports both "2026 Cup Season" and "2026 Season" headings;
- retains V2.12 component normalization and all existing model weights unchanged.

Expected: Checkpoints 1 and 2 show populated component scores; Checkpoint 3 shows positive
and negative Reference z-scores.
