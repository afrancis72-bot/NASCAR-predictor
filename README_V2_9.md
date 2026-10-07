# NASCAR Predictor V2.9 — Driver DNA Integration Audit

V2.9 moves NASCAR Reference from a display layer into the simulation feature pipeline.

Key changes:
- `comparable_track_avg_finish` maps to the generic `track_type_avg_finish` engine input.
- Current race strength uses season finish, recent finish, Reference recent-form score,
  Reference place-differential score, Reference reliability, and DK baseline.
- Track-type fit uses comparable-track finish and track-type Elo.
- NASCAR Reference Advanced Metrics Reliability is parsed when present; missing remains neutral.
- Driver DNA page includes a Model Integration Audit showing the normalized Reference signals
  that actually enter race strength and Track DNA fit.
- Obsolete V2.7 "Extracted raw values" table is removed from the diagnostic view.
- Final GPP remains locked until the complete official qualifying grid is validated.

For Atlanta, comparable-track data is the source's Superspeedway profile.
