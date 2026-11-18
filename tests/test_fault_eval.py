import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from upkeep.fault_eval import evaluate_faults


class FaultEvaluationTest(TestCase):
    def test_uses_held_out_window_and_matching_turbine(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cln").mkdir()
            (root / "fault_events.csv").write_text(
                "dataset_id,turbine_id,fault_type,fault_timestamp,event_start_id\n"
                "A-7,A-T2,Gearbox failure,2020-01-08 00:00:00,60\n",
                encoding="utf-8",
            )
            timestamps = pd.date_range("2023-01-01", periods=6)
            pd.DataFrame({
                "row_id": [10, 20, 30, 40, 50, 60],
                "turbine_id": 2,
                "timestamp": timestamps,
                "split": ["prediction", "prediction", "train", "prediction", "prediction", "prediction"],
                "status_id": 0,
            }).to_csv(root / "cln" / "7_cln.csv", sep=";", index=False)
            health_path = root / "health_A-7.jsonl"

            def record(row_id, health, turbine="A-T2"):
                return {
                    "row_id": row_id,
                    "dataset_id": "A-7",
                    "turbine_id": turbine,
                    "timestamp": timestamps[row_id // 10 - 1].isoformat(),
                    "health": health,
                }

            def evaluate(records):
                health_path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
                return evaluate_faults(root, root, pd.Timedelta(days=4)).iloc[0]

            excluded = [record(10, "red"), record(30, "red"), record(60, "red"), record(20, "red", "A-T3")]
            result = evaluate(excluded + [record(50, "red"), record(40, "green"), record(20, "yellow")])
            self.assertEqual(result["eligible_readings"], 3)
            self.assertEqual(result["first_warning_timestamp"], timestamps[1])
            self.assertEqual(result["first_warning_health"], "yellow")
            self.assertEqual(result["lead_time_hours"], 96)
            self.assertEqual(result["status"], "complete")

            result = evaluate(excluded + [record(40, "green"), record(50, None)])
            self.assertTrue(pd.isna(result["warning_detected"]))
            self.assertEqual(result["scored_readings"], 1)
            self.assertEqual(result["status"], "partial_predictions")

            result = evaluate(excluded + [record(row, "green") for row in (20, 40, 50)])
            self.assertFalse(result["warning_detected"])
            self.assertEqual(result["status"], "complete")

            health_path.unlink()
            result = evaluate_faults(root, root, pd.Timedelta(days=4)).iloc[0]
            self.assertEqual(result["status"], "missing_predictions")
            self.assertTrue(pd.isna(result["warning_detected"]))
