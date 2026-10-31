"""Regression tests for NASCAR driver/scenario alignment and track-family inputs.

Run: python -m unittest discover -s tests -p 'test_nascar_integrity.py' -v
These use synthetic pre-race data; they are not historical profitability backtests.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "nascar_engine" if (ROOT / "nascar_engine").is_dir() else ROOT
sys.path.insert(0, str(ENGINE))

from nascar_predictor_pro import Config, build_features, simulate
from nascar_reference_dna import comparable_track_family


class NascarIntegrityTests(unittest.TestCase):
    def test_every_catalog_track_has_an_explicit_family(self):
        catalog = pd.read_csv(ENGINE / "track_catalog.csv")
        categories = {"short_track", "road_course", "intermediate", "superspeedway"}
        for name in catalog["track"]:
            with self.subTest(track=name):
                self.assertIn(comparable_track_family(name), categories)
        self.assertEqual(comparable_track_family("Martinsville Speedway"), "short_track")
        self.assertEqual(comparable_track_family("Bristol Motor Speedway"), "short_track")
        self.assertEqual(comparable_track_family("Richmond Raceway"), "short_track")
        self.assertEqual(comparable_track_family("Phoenix Raceway"), "short_track")
        self.assertEqual(comparable_track_family("Watkins Glen International"), "road_course")
        self.assertEqual(comparable_track_family("Circuit of the Americas"), "road_course")
        self.assertEqual(comparable_track_family("Daytona International Speedway"), "superspeedway")
        self.assertEqual(comparable_track_family("Talladega Superspeedway"), "superspeedway")
        self.assertEqual(comparable_track_family("Atlanta Motor Speedway"), "superspeedway")
        self.assertEqual(comparable_track_family("Kansas Speedway"), "intermediate")
        with self.assertRaises(ValueError):
            comparable_track_family("Unknown Speedway")

    def test_ranked_simulation_columns_retain_driver_identity(self):
        names = [f"Driver {j}" for j in range(8)]
        dk = pd.DataFrame({
            "Name": names,
            "ID": [str(1000+j) for j in range(8)],
            "Salary": [7000, 9900, 7200, 9600, 7600, 9100, 6800, 8800],
            "AvgPointsPerGame": [12, 65, 19, 57, 24, 50, 10, 45],
        })
        priors = pd.DataFrame({"Name": names,
                               "manual_speed_rating": [10, 95, 20, 88, 30, 75, 5, 65]})
        catalog = pd.read_csv(ENGINE / "track_catalog.csv")
        track = catalog[catalog["track"].eq("Martinsville Speedway")].copy()
        grid = pd.DataFrame({"Name": names, "qualifying_position": [8, 1, 7, 2, 6, 3, 5, 4]})
        feature = build_features(dk, priors, track, grid)
        # Force the source order to be opposite strength order: the old bug
        # incorrectly left matrix columns in this order after projections sorted.
        feature = feature.sort_values("race_strength", ascending=True).reset_index(drop=True)
        cfg = Config(sims=400, seed=144, laps=500)
        projected, matrix = simulate(feature, cfg)
        self.assertEqual(matrix.shape, (400, 8))
        self.assertNotEqual(projected["Name"].tolist(), feature["Name"].tolist())
        np.testing.assert_allclose(projected["proj_dk"], matrix.mean(axis=0), atol=0.02)
        np.testing.assert_allclose(projected["ceiling_p90"], np.quantile(matrix, .90, axis=0), atol=0.02)
        np.testing.assert_allclose(projected["ceiling_p99"], np.quantile(matrix, .99, axis=0), atol=0.02)
        self.assertTrue(projected["proj_dk"].is_monotonic_decreasing)
        # In every simulation, exactly 500 laps led and 500 fastest laps
        # must be distributed across the same eight drivers.
        self.assertAlmostEqual(float(projected["exp_laps_led"].sum()), 500, places=3)
        self.assertAlmostEqual(float(projected["exp_fastest_laps"].sum()), 500, places=3)


if __name__ == "__main__":
    unittest.main()
