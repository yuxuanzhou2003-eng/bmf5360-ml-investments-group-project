import unittest

import numpy as np
import pandas as pd

import build_panel_v2 as panel


class PanelV2FocusedTests(unittest.TestCase):
    def test_exact_membership_is_inclusive_at_both_interval_boundaries(self):
        intervals = pd.DataFrame({"ric": ["SRC", "RCV"],
                                  "start": pd.to_datetime(["2026-06-30", "2026-06-29"]),
                                  "end": pd.to_datetime(["2026-06-30", "2026-06-30"])})
        members_on = panel.membership_lookup(intervals)
        self.assertEqual(members_on("2026-06-30"), {"SRC", "RCV"})
        self.assertEqual(members_on("2026-07-01"), set())

    def test_event_end_includes_all_times_on_the_configured_final_day(self):
        actuals = pd.DataFrame({"announcement": pd.to_datetime(["2026-06-30 16:30", "2026-07-01 00:00"]),
                                "Instrument": ["SRC", "OUT"]})
        selected = panel.event_window(actuals, {"event_start": "2015-01-01", "event_end": "2026-06-30"})
        self.assertEqual(selected.Instrument.tolist(), ["SRC"])

    def test_pairwise_beta_uses_only_the_asset_benchmark_overlap(self):
        index = pd.bdate_range("2024-01-01", periods=126)
        market = pd.Series(np.linspace(-0.03, 0.03, len(index)), index=index)
        asset = 4 * market + 0.01
        asset.iloc[:26] = np.nan
        self.assertAlmostEqual(panel.pairwise_beta(asset, market, 100), 4.0, places=12)

    def test_graph_history_excludes_the_announcement_day_and_is_limited_to_126_sessions(self):
        index = pd.bdate_range("2024-01-01", periods=127)
        market = np.linspace(-0.02, 0.02, len(index))
        wide = pd.DataFrame({"SPY.P": market,
                             "SRC": 1.5 * market + np.sin(np.arange(len(index))) / 100,
                             "RCV": 0.7 * market + np.cos(np.arange(len(index))) / 100}, index=index)
        _, last_day = panel.residual_correlations(wide, index[-1], 126, 100, "SPY.P")
        self.assertEqual(last_day, index[-2])

    def test_nonmember_is_excluded_before_top_k_selection(self):
        correlation = pd.Series({"SRC": 1.0, "OUT": 0.95, "IN": 0.60})
        selection = {"mode": "topk_with_threshold", "max_neighbors": 1,
                     "min_abs_correlation": 0.3, "positive_only": False}
        selected = panel.pick_neighbours(correlation, "SRC", {"SRC", "IN"}, selection)
        self.assertEqual(selected.index.tolist(), ["IN"])

    def test_missing_future_window_returns_an_incomplete_label_record(self):
        sessions = pd.bdate_range("2026-06-29", periods=3)
        wide = pd.DataFrame({"SPY.P": [0.01, 0.01, 0.01], "RCV": [0.02, 0.02, 0.02]}, index=sessions)
        label = panel.forward_label(wide, sessions, 1, 5, "RCV", "SPY.P")
        self.assertFalse(label["label_complete"])
        self.assertTrue(pd.isna(label["exit_close"]))

    def test_diagnostics_see_receiver_actuals_after_configured_event_end(self):
        actual_days = {"RCV": pd.DatetimeIndex([pd.Timestamp("2026-07-02")])}
        diag = panel.receiver_diagnostics(actual_days, "RCV", "2026-06-30", pd.Timestamp("2026-07-03"))
        self.assertTrue(diag["receiver_event_coverage_present"])
        self.assertTrue(diag["ex_post_own_announcement_overlap"])

    def test_diagnostics_do_not_treat_empty_own_actuals_as_no_overlap(self):
        diag = panel.receiver_diagnostics({}, "RCV", "2026-06-30", pd.Timestamp("2026-07-03"))
        self.assertFalse(diag["receiver_event_coverage_present"])
        self.assertIsNone(diag["ex_post_own_announcement_overlap"])


if __name__ == "__main__":
    unittest.main()
