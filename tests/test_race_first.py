"""Regression tests for the race-first simulation contract."""
import unittest
import numpy as np
import pandas as pd
from nascar_race_engine import simulate
from nascar_predictor_pro import Config


class RaceFirstTests(unittest.TestCase):
    def setUp(self):
        n=12
        self.features=pd.DataFrame({
            "Name":[f"Driver {i}" for i in range(n)],
            "ID":[str(i) for i in range(n)],
            "Salary":[8000]*n,
            "AvgPointsPerGame":[30.0]*n,
            "race_strength":np.linspace(1,-1,n),
            "dominator_strength":np.linspace(1,-1,n),
            "projected_start":np.arange(1,n+1),
        })
        for key in ("data_coverage","model_confidence","place_diff_room","front_start_score",
                    "back_start_score","audit_dk_prior","audit_vegas","audit_current",
                    "audit_intermediate","audit_reference_dna","audit_reference_form",
                    "audit_reference_pd","audit_reference_track","audit_reference_elo",
                    "audit_reference_reliability"):
            self.features[key]=.5

    def test_reproducible_and_aligned(self):
        cfg=Config(sims=120,seed=23,laps=50)
        a,m=simulate(self.features,cfg)
        b,m2=simulate(self.features,cfg)
        np.testing.assert_array_equal(m,m2)
        np.testing.assert_array_equal(a["ID"],b["ID"])
        np.testing.assert_allclose(m.mean(axis=0),a["proj_dk"],atol=.001)
        self.assertAlmostEqual(float(a["exp_laps_led"].sum()),50,places=5)
        self.assertAlmostEqual(float(a["exp_fastest_laps"].sum()),50,places=5)
        self.assertAlmostEqual(float(a["win_pct"].sum()),1,places=5)

    def test_invalid_start_grid(self):
        f=self.features.copy()
        f.loc[0,"projected_start"]=2
        with self.assertRaises(ValueError):
            simulate(f,Config(sims=10,laps=50))


if __name__=="__main__":
    unittest.main()
