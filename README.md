# NASCAR Predictor V2.12

## Component Wiring Fix
V2.11 proved the break was inside `build_features()`: the overall Driver DNA score
survived, but valid scored component columns were being neutralized.

V2.12:
- normalizes the proven scored Reference component columns directly;
- retains neutral handling for genuinely missing/constant fields;
- uses observed raw-data fallbacks only when a scored component has no spread;
- keeps the three-checkpoint pipeline validation;
- requires at least 3 of 5 component signals to show cross-driver spread for PASS;
- does not change model weights;
- keeps final simulation caution until integration validates.

Repository package is cleaned to one README.
