# NASCAR Predictor V2.6 — NASCAR Reference Parser Fix
- Correctly parses race-level Recent Results (including ordinal finishes such as 1st).
- Parses Best Tracks for track-history average finish.
- For modern Atlanta, uses the source's Superspeedway performance profile as comparable-track evidence.
- Reads source Reliability when present.
- Removes the unsupported Reference dominator component rather than fabricating it.
- Uses one coverage/confidence system in the Driver DNA screen.
- Warns when median source coverage is below 55%.
- Missing data remains neutral and available weights renormalize.
