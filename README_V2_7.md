# NASCAR Predictor V2.7 — Extraction Diagnostic

No Driver DNA model-weight changes.

Adds an Extraction Diagnostic tab that:
- selects any DK driver
- shows the raw values currently extracted for that driver
- fetches the actual NASCAR Reference driver page
- displays every HTML table found, including normalized column names
- shows a text sample of the returned page

Purpose: identify exactly why NASCAR Reference connectivity succeeds while statistical fields remain empty before changing the parser again.
