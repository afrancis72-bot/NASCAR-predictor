# NASCAR Predictor V2.8 — Text Parser

NASCAR Reference returns full statistical page content but no conventional HTML tables.
V2.8 therefore parses the rendered statistical text sections directly.

Atlanta Driver DNA now uses:
- 2026 season average finish
- last-5 / recent Cup form
- recent start-to-finish place differential
- Superspeedway average finish
- Superspeedway NR-Rating Elo
- consistency/reliability proxy
- DK market baseline

The Extraction Diagnostic now displays the V2.8 parser output and parsed recent Cup races.
Missing fields remain neutral; weights renormalize.
