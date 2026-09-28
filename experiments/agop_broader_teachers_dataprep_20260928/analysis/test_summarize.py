"""Synthetic bookkeeping checks only; no numerical engine or array imports."""
import json
from pathlib import Path
import tempfile
import unittest

import summarize


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = []
        for number in range(28):
            engine = "swiglu" if number in (2, 3) else "relu"
            cid = f"arm{number}"
            args = {"teacher": "h3", "seed": number + 501, "r": 8, "d": 64,
                    "m": 128, "scale": .0001, "steps": 20000, "id": cid}
            if engine == "swiglu":
                args = [{"tag": cid, "rank": 8, "args": {"link": "he3", "seed": number + 501,
                        "d": 64, "m": 64, "s": .05, "c": [1] * 8, "max_steps": 1000}}]
            self.entries.append({"id": cid, "engine": engine, "config": args})
        self.manifest = {"configs": self.entries, "protocol_version": "synthetic", "runtime": {}}
        self.write("MANIFEST.snapshot.json", self.manifest)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_all_28_and_copied_events_separate_from_censoring(self):
        self.write("STATUS.json", {"status": "capped", "pending": [f"arm{i}" for i in range(5, 28)],
            "completed": [{"id": f"arm{i}", "returncode": 0, "engine": self.entries[i]["engine"]} for i in range(5)], "active": []})
        self.write("data/arm0/DONE.json", {"joint_observed": True, "joint_status": "observed", "first_joint_step": 10,
            "training_censored": True, "diagnostic_censored": True, "plateau_end": 12, "observed_steps": 20})
        rows = [{"step": 0, "A_min": .1}, {"step": 10, "A_min_gain": .6, "in_plateau": True, "refit_gain_interval": [.5, .6]}]
        (self.root / "data/arm0/rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        self.write("data/arm1/DONE.json", {"complete": True, "training_complete": True, "diagnostic_complete": True,
            "joint_observed": False, "joint_status": "not_observed_resolved_grid"})
        for number, reason in ((2, "max_steps"), (3, "loss_stop")):
            self.write(f"data/arm{number}/result.json", {"tag": f"arm{number}", "rank": 8,
                "cfg": self.entries[number]["config"][0]["args"], "status": "loss_stop" if number == 3 else "censored_or_failed",
                "termination": {"reason": reason, "step": 1000}, "completed": True})
        summary = summarize.build_summary(self.root)
        self.assertEqual(summary["reported_arms"], 28)
        self.assertEqual(sum(summary["outcome_counts"].values()), 28)
        records = summary["records"]
        self.assertEqual([row["outcome_category"] for row in records[:5]],
                         ["censored", "planned_horizon_complete", "censored", "loss_target_stop", "failed_missing_completion_indicator"])
        self.assertEqual(summary["outcome_counts"]["not_started"], 23)
        joint = records[0]["scientific"]["joint_in_own_plateau"]
        self.assertTrue(joint["joint_observed"])
        self.assertEqual(joint["first_joint_checkpoint"]["step"], 10)
        self.assertIsNone(records[2]["scientific"]["joint_in_own_plateau"]["joint_observed"])
        self.assertIsNone(records[2]["scientific"]["plateau"]["plateau_right_censored"])
        json.dumps(summary, allow_nan=False)

    def test_history_fallback_keeps_last_valid_receipt(self):
        (self.root / "STATE_HISTORY.jsonl").write_text(json.dumps({"status": "running", "active": [{"id": "arm0"}]}) + "\n{broken\n")
        summary = summarize.build_summary(self.root)
        self.assertEqual(summary["records"][0]["outcome_category"], "active_snapshot")
        self.assertEqual(summary["launcher_status_source"], "STATE_HISTORY.jsonl:last_parseable_object")
        self.assertTrue(summary["issues"])

    def test_invalid_and_nonfinite_records_do_not_remove_arms(self):
        self.write("STATUS.json", {"status": "completed", "completed": [{"id": "arm0", "returncode": 0}]})
        directory = self.root / "data/arm0"
        directory.mkdir(parents=True)
        (directory / "DONE.json").write_text('{"complete":true,"complete":false}')
        (directory / "rows.jsonl").write_text('{"step":0,"A_min":NaN}\n')
        summary = summarize.build_summary(self.root)
        self.assertEqual(summary["reported_arms"], 28)
        self.assertEqual(summary["records"][0]["outcome_category"], "failed_unreadable_completion_indicator")
        self.assertIsNone(summary["records"][0]["scientific"]["diagnostics"]["initial"]["A_min"])
        self.assertGreaterEqual(len(summary["issues"]), 2)
        json.dumps(summary, allow_nan=False)

    def test_dispatch_cli_writes_three_companions(self):
        out = self.root / "derived/summary.json"
        self.assertEqual(summarize.main(["--execution", str(self.root), "--out", str(out)]), 0)
        self.assertEqual(len(json.loads(out.read_text())["records"]), 28)
        self.assertTrue(out.with_suffix(".csv").is_file())
        self.assertTrue(out.with_suffix(".md").is_file())


if __name__ == "__main__":
    unittest.main()
