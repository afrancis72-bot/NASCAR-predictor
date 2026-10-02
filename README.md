# NASCAR Predictor V1.4

A CSV-driven NASCAR Cup DFS model built to mirror the architecture of the existing PGA predictor:

**Track DNA → Driver Strength → Monte Carlo → DraftKings Projection → Optimizer**

## Current build
V1A is a **pre-practice / pre-qualifying** model for the 2026 South Point 400 at Las Vegas.
It uses the uploaded DraftKings field and salaries, current DK points baseline, 2026 1.5-mile evidence,
Las Vegas spring evidence, selected season/venue priors, and a separate dominator model.

Missing specialized statistics are shrunk to field-neutral rather than treated as zero.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Saturday update
1. Open the **Live Update** tab.
2. Download `practice_qualifying_template.csv`.
3. Enter NASCAR practice ranks (single lap and 5/10/15/20-lap averages where available) and qualifying position.
4. Upload the completed CSV.
5. Re-run projections and DFS optimizer.

The model weights longer-run practice more heavily than single-lap speed.

## DraftKings scoring implemented
- Place differential: +/- 1 point
- Fastest laps: +0.45
- Laps led: +0.25
- Finishing-position points: current DraftKings NASCAR table

## Important V1 limitation
This is the architecture + first live test, not a fully trained historical model yet.
Track DNA weights are transparent hand-set priors informed by current Las Vegas / 1.5-mile evidence.
The next major step is to ingest historical race-level loop/practice/qualifying data and learn/calibrate
track-archetype weights using walk-forward backtesting.

## Files
- `app.py` — Streamlit UI
- `nascar_predictor_pro.py` — features, Monte Carlo, DK scoring, optimizer
- `data/dk_salaries_las_vegas_2026.csv` — uploaded DK slate
- `data/driver_priors_las_vegas_2026.csv` — V1A driver priors
- `data/track_profiles.csv` — Track DNA
- `data/practice_qualifying_template.csv` — Saturday update template


## V1.1 upgrade
Ceiling optimization now evaluates six-driver rosters within the same Monte Carlo race outcomes. Added complete 2026 spring Las Vegas finishing/start/laps-led evidence for the current DK field where available.

## V1.2 intelligence upgrade
- Expanded Next Gen 1.5-mile dominator priors.
- Added current speed / Chase-form evidence where supported.
- Reliability-aware uncertainty: incomplete drivers get wider simulation distributions instead of false certainty.
- Starting-grid architecture: qualifying position drives place-differential room and a modest front-start dominator effect.
- Diagnostics tab with model confidence and optimizer exposure.
- Saturday update remains one CSV upload; do not confuse qualifying order with official starting position.

## V1.3 scenario-aware portfolio
The ceiling/GPP optimizer now classifies coherent Monte Carlo races into Hamlin-dominant,
Larson-dominant, split-dominator, chaos/attrition, and track-position scripts. Their weights
are learned from simulation frequency. Candidate lineups are scored across those scripts and
selected for marginal scenario coverage subject to the user's overlap setting. No driver is
forced into the portfolio and no manual exposure cap is used.

## V1.4 calibration fix
- Sparse optional statistics require at least six real observations before influencing a driver.
- Missing sparse evidence is neutral rather than median-imputed into tiny samples.
- Sparse z-scores are clipped and reliability-shrunk.
- DraftKings fantasy baseline is now a weak prior rather than a dominant/double-counted signal.
- Diagnostics exposes DK prior, Vegas, current-form, and intermediate components for auditing.
