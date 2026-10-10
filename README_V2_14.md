# NASCAR Predictor V2.14

V2.14 hardens the V2.13 race simulation and scenario portfolio builder after a low-data deep-starter audit.

## Changes
- Low-confidence/low-coverage driver strength is conservatively shrunk toward the observable DK market prior.
- The uncertainty variance bonus is capped so missing data does not itself create excessive DFS ceiling.
- Automatic driver-specific portfolio caps: LOW/thin data max 30%; MEDIUM max 45%; user max exposure remains the absolute ceiling.
- Scenario portfolio viability gate requires at least 82% of best candidate simulated mean and 85% of best candidate raw GPP score.
- Seed/fallback lineups that fail viability are replaced before scenario optimization.
- Every final scenario lineup receives Mean, P10, P90, P95, GPP_Score, ScenarioScore and viability thresholds.

No driver is named or manually excluded. Rules are data-driven and tournament-agnostic.
