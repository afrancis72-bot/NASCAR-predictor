# NASCAR Predictor V2.5 — NASCAR Reference Driver DNA

V2.5 replaces the blocked NASCAR.com scraper with NASCAR Reference, which passed the deployed Streamlit diagnostic.

Driver DNA components:
- recent Cup form
- recent place differential
- track-history finish performance
- track-history laps-led / dominator proxy
- reliability proxy
- DraftKings market baseline

Missing components are omitted and weights renormalize; they are never scored as zero.
Before qualifying, the Driver DNA table displays PRE-QUAL rather than a fake starting position.
