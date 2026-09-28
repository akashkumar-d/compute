#!/usr/bin/env python3
"""Summarize declared arms and saved execution records, using only the stdlib.

No model imports, array loading, training, moment evaluation, interpolation,
new event thresholds, or scientific certification are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TERMINAL_LAUNCHER = {"completed", "failed", "capped", "queue_timeout", "interrupted"}
INDICATORS = {"relu": "DONE.json", "swiglu": "result.json"}
JOINT_FIELDS = (
    "joint_observed", "joint_status", "primary_event_status", "first_joint_step",
    "max_resolved_plateau_q", "full_grid_event_negative_resolved",
)
RELU_METRICS = (
    "step", "loss", "time_eta", "in_plateau", "A_top", "A_min", "A_mean",
    "A_min_gain", "A_mean_gain", "refit", "refit_lower", "refit_gap",
    "refit_gain_interval", "q_reference", "q_reference_resolved", "agop_valid",
    "relative_gap", "relative_lambda_r", "agop_eigen_residual",
    "agop_teacher_energy_fraction", "balanced_weak_direction_energy",
    "principal_cosines_sq_ascending", "teacher_direction_captures",
    "refit_info", "teacher_mean", "target_linear_norm_sq", "target_nonlinear_variance",
    "mean_error_sq", "linear_error_sq", "loss_nonlinear", "loss_nonlinear_relative",
)
SWIGLU_METRICS = (
    "index", "step", "t", "status", "error", "original_L", "L",
    "agop_Atop", "agop_Amin", "agop_A", "agop_full_rank_resolved",
    "agop_leading_resolved", "agop_psd_resolved", "agop_resolution_guard",
    "agop_rank_r_relative_gap", "agop_rank_r_eigenvalue_relative",
    "agop_teacher_trace_energy_fraction", "agop_balanced_weakest_direction_energy",
    "agop_principal_cosines_squared_descending", "agop_teacher_axis_captures",
    "pinv", "ridge_actual", "refit_source", "equilibrated_lambda_ratio",
    "normalization_V",
)


def subset(value, keys):
    return {key: value.get(key) for key in keys} if isinstance(value, dict) else None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Reader:
    """Read only named JSON records; retain parse errors and byte provenance."""
    def __init__(self):
        self.sources = {}
        self.issues = []

    def issue(self, path, message):
        self.issues.append({"path": str(path), "message": message})

    def decode(self, text, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def nonfinite(token):
            self.issue(path, f"nonfinite JSON constant {token} replaced with null")
            return None

        value = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)

        def finite(item):
            if isinstance(item, float) and not math.isfinite(item):
                self.issue(path, "overflowing JSON number replaced with null")
                return None
            if isinstance(item, dict):
                return {key: finite(val) for key, val in item.items()}
            if isinstance(item, list):
                return [finite(val) for val in item]
            return item
        return finite(value)

    def read(self, path, *, lines=False):
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = path.read_bytes()
            self.sources[str(path)] = {"path": str(path), "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()}
            text = data.decode("utf-8")
            if not lines:
                return self.decode(text, path)
            rows = []
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(self.decode(line, path))
                except (ValueError, TypeError) as exc:
                    self.issue(path, f"line {number}: {exc}; line unavailable")
            return rows
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            self.issue(path, f"cannot read record: {type(exc).__name__}: {exc}")
            return None

    def object(self, path):
        result = self.read(path)
        if result is not None and not isinstance(result, dict):
            self.issue(path, "expected JSON object")
            return None
        return result


def flag_counts(rows, key):
    return {"true": sum(row.get(key) is True for row in rows),
            "false": sum(row.get(key) is False for row in rows),
            "unavailable": sum(row.get(key) is not True and row.get(key) is not False for row in rows)}


def indexed_row(rows, key, value, reader, path):
    if value is None:
        return None
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) > 1:
        reader.issue(path, f"duplicate {key}={value}; selected checkpoint unavailable")
        return None
    return matches[0] if matches else None


def declared(entry):
    cfg = entry["config"]
    args = cfg if entry["engine"] == "relu" else cfg[0]["args"]
    return {"teacher": args.get("teacher", args.get("link")), "seed": args.get("seed"),
            "rank": args.get("r", cfg[0].get("rank") if isinstance(cfg, list) else None),
            "dimension": args.get("d"), "width": args.get("m"),
            "initial_scale": args.get("scale", args.get("s")),
            "planned_updates": args.get("steps", args.get("max_steps"))}


def validate_record_config(actual, expected, reader, path, *, extras=False):
    if actual is None:
        return None
    if not isinstance(actual, dict):
        reader.issue(path, "recorded config is not an object")
        return False
    matches = all(key in actual and actual[key] == val for key, val in expected.items())
    if not extras:
        matches = matches and set(actual) == set(expected)
    if not matches:
        reader.issue(path, "recorded config does not match declared arm; do not use as matching scientific evidence")
    return matches


def relu_record(directory, entry, reader):
    done = reader.object(directory / "DONE.json")
    train = reader.object(directory / "TRAIN_RESULT.json")
    meta = reader.object(directory / "meta.json")
    failed = reader.object(directory / "FAILED.json")
    loaded_rows = reader.read(directory / "rows.jsonl", lines=True)
    rows = [row for row in loaded_rows or [] if isinstance(row, dict)]
    if loaded_rows is not None and len(rows) != len(loaded_rows):
        reader.issue(directory / "rows.jsonl", "non-object diagnostic rows omitted")
    identity = validate_record_config(meta.get("config") if meta else None,
                                     entry["config"], reader, directory / "meta.json")
    if done and done.get("config_sha256") is not None:
        expected = hashlib.sha256(canonical(entry["config"]).encode()).hexdigest()
        if done["config_sha256"] != expected:
            identity = False
            reader.issue(directory / "DONE.json", "config_sha256 differs from declared configuration")
    endpoint = done.get("plateau_end") if done else train.get("plateau_end") if train else None
    initial = indexed_row(rows, "step", 0, reader, directory / "rows.jsonl")
    endrow = indexed_row(rows, "step", endpoint, reader, directory / "rows.jsonl")
    first = indexed_row(rows, "step", done.get("first_joint_step") if done else None,
                        reader, directory / "rows.jsonl")
    joint = {key: done.get(key) if done else None for key in JOINT_FIELDS}
    joint.update(source="DONE.json" if done else None,
                 availability="recorded" if done and any(key in done for key in JOINT_FIELDS) else "not_recorded",
                 interpretation="Copied legacy same-saved-checkpoint, own-initial-plateau event; heuristic spectral flags are not certificates.",
                 first_joint_checkpoint=subset(first, RELU_METRICS))
    if done and train:
        for key in ("observed_steps", "planned_steps", "plateau_end", "training_censored", "plateau_right_censored", "final_loss"):
            if key in done and key in train and done[key] != train[key]:
                reader.issue(directory, f"DONE.json and TRAIN_RESULT.json disagree on {key}")
    return {
        "record_present": done is not None, "identity_matches_manifest": identity,
        "engine_status_recorded": None,
        "termination": {
            "training_stop_reason": done.get("training_stop_reason") if done else train.get("stop_reason") if train else None,
            "diagnostic_stop_reason": done.get("diagnostic_stop_reason") if done else None,
            "observed_steps": done.get("observed_steps") if done else train.get("observed_steps") if train else None,
            "planned_steps": done.get("planned_steps") if done else train.get("planned_steps") if train else None,
            "final_loss": done.get("final_loss") if done else train.get("final_loss") if train else None,
            "training_complete": done.get("training_complete") if done else train.get("complete") if train else None,
            "training_censored": done.get("training_censored") if done else train.get("training_censored") if train else None,
            "diagnostic_complete": done.get("diagnostic_complete") if done else None,
            "diagnostic_censored": done.get("diagnostic_censored") if done else None,
            "complete": done.get("complete") if done else None,
            "elapsed_seconds": done.get("elapsed_seconds") if done else train.get("elapsed_seconds") if train else None,
        },
        "failure": failed,
        "plateau": {
            "rule": meta.get("plateau_rule") if meta else "All-update initial prefix in [.95,1.05] with running max/min<=1.05",
            "units": "raw population MSE",
            "last_admissible_update": endpoint,
            **{key: done.get(key) if done else train.get(key) if train else None for key in (
                "plateau_right_censored", "plateau_exit_observed", "true_plateau_endpoint_diagnosed",
                "endpoint_is_censored_boundary", "prefix_diagnostics_complete")},
        },
        "joint_in_own_plateau": joint,
        "diagnostics": {"recorded_rows": len(rows) if loaded_rows is not None else None,
                        "required_steps": done.get("required_diagnostic_steps") if done else None,
                        "recorded_steps": [row.get("step") for row in rows] if loaded_rows is not None else None,
                        "agop_valid_recorded_flags": flag_counts(rows, "agop_valid") if loaded_rows is not None else None,
                        "initial": subset(initial, RELU_METRICS),
                        "primary_prefix_endpoint": subset(endrow, RELU_METRICS)},
        "recorded_endpoint_fields": subset(done, (
            "initial_A_min", "initial_refit", "endpoint_observed", "endpoint_A_min", "endpoint_A_min_gain",
            "endpoint_A_mean", "endpoint_refit", "endpoint_refit_gain", "endpoint_agop_valid")),
        "refit_threshold_eligibility": None,
        "refit_threshold_eligibility_note": "Not separately recorded by this driver; no new threshold inference is made.",
    }


def swiglu_record(directory, entry, reader):
    result = reader.object(directory / "result.json")
    launch = reader.object(directory / "launch.json")
    raw_path = directory / (entry["id"] + ".json")
    if not raw_path.is_file():
        raw_path = directory / "partials" / (entry["id"] + ".json")
    raw = reader.object(raw_path)
    partial = reader.object(directory / "diagnostics_partial.json") if result is None else None
    expected = entry["config"][0]["args"]
    header = result or launch
    identity = validate_record_config(header.get("cfg") if header else None, expected,
                                     reader, directory / ("result.json" if result else "launch.json"))
    if header and (header.get("tag") != entry["id"] or header.get("rank") != entry["config"][0]["rank"]):
        identity = False
        reader.issue(directory, "recorded tag/rank does not match declared arm")
    if raw:
        raw_match = validate_record_config(raw.get("cfg"), expected, reader, raw_path, extras=True)
        if raw_match is False:
            identity = False
    diagnostics = result.get("diagnostics") if result else partial.get("rows") if partial else None
    rows = [row for row in diagnostics or [] if isinstance(row, dict)] if isinstance(diagnostics, list) else []
    if diagnostics is not None and (not isinstance(diagnostics, list) or len(rows) != len(diagnostics)):
        reader.issue(directory, "invalid diagnostic row list; affected rows unavailable")
    termination = result.get("termination") if result else raw.get("termination") if raw else None
    termination = termination if isinstance(termination, dict) else None
    prefixes = result.get("prefixes") if result else None
    primary = prefixes.get("0.001") if isinstance(prefixes, dict) else None
    checkpoint = primary.get("checkpoint_index") if isinstance(primary, dict) else None
    initial = indexed_row(rows, "step", 0, reader, directory)
    endpoint = indexed_row(rows, "index", checkpoint, reader, directory)
    terminal = indexed_row(rows, "step", termination.get("step") if termination else None, reader, directory)
    if result and raw and isinstance(raw.get("termination"), dict) and termination:
        for key in ("step", "t", "L", "reason"):
            if raw["termination"].get(key) != termination.get(key):
                reader.issue(directory, f"result/raw termination mismatch for {key}")
    joint = {key: result.get(key) if result else None for key in JOINT_FIELDS}
    joint.update(source="result.json" if result and any(key in result for key in JOINT_FIELDS) else None,
                 availability="recorded" if result and any(key in result for key in JOINT_FIELDS) else "not_recorded",
                 interpretation="The existing SwiGLU driver does not persist a joint-event classification; null is unavailable, not a negative event.",
                 first_joint_checkpoint=None)
    return {
        "record_present": result is not None, "identity_matches_manifest": identity,
        "engine_status_recorded": result.get("status") if result else None,
        "termination": {"engine_termination": termination,
                        "wall_stop_cause": result.get("wall_stop_cause") if result else None,
                        "completed_flag": result.get("completed") if result else None,
                        "raw_partial": result.get("raw_partial") if result else raw.get("partial") if raw else None,
                        "effective_training_wall_seconds": header.get("effective_training_wall_seconds") if header else None,
                        "training_wall_seconds": result.get("training_wall_seconds") if result else None,
                        "diagnostic_wall_seconds": result.get("diagnostic_wall_seconds") if result else None,
                        "wall_seconds": result.get("wall_seconds") if result else None},
        "failure": {"training_exception": result.get("training_exception")} if result and result.get("training_exception") is not None else None,
        "plateau": {"rule": "All-update absolute change from initial normalized loss <=0.001; secondary <=0.0001",
                    "units": "population MSE divided by teacher variance",
                    "prefixes_recorded": prefixes,
                    "last_admissible_update": primary.get("last_admissible_update") if isinstance(primary, dict) else None,
                    "plateau_right_censored": result.get("prefix_right_censored") if result else None,
                    "censoring_note": "The driver records prefix endpoints but no explicit prefix-censoring flag; no recomputation is made."},
        "joint_in_own_plateau": joint,
        "diagnostics": {"recorded_rows": len(rows) if isinstance(diagnostics, list) else None,
                        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())) if isinstance(diagnostics, list) else None,
                        "checkpoint_count_recorded": result.get("checkpoint_count") if result else None,
                        "success_count_recorded": result.get("diagnostic_success_count") if result else None,
                        "full_rank_resolution_recorded_flags": flag_counts(rows, "agop_full_rank_resolved") if isinstance(diagnostics, list) else None,
                        "leading_resolution_recorded_flags": flag_counts(rows, "agop_leading_resolved") if isinstance(diagnostics, list) else None,
                        "initial": subset(initial, SWIGLU_METRICS),
                        "primary_prefix_last_saved": subset(endpoint, SWIGLU_METRICS),
                        "terminal": subset(terminal, SWIGLU_METRICS),
                        "quadrature_double_recorded": result.get("quadrature_double") if result else None},
        "normalization_recorded": subset(raw, ("gamma", "EY", "V")),
        "all_update_count_recorded": result.get("all_update_count") if result else None,
        "raw_record_path": str(raw_path) if raw else None,
    }


def classify(record):
    execution, scientific = record["execution"], record["scientific"]
    state = execution["state"]
    if state == "not_started":
        return "not_started"
    if state == "active":
        return "active_snapshot"
    receipt = execution["completion_receipt"]
    if receipt is not None:
        if not execution["completion_indicator_exists"]:
            return "failed_missing_completion_indicator"
        if not scientific["record_present"]:
            return "failed_unreadable_completion_indicator"
        if receipt.get("returncode") not in (0, None) and not receipt.get("stop_reason"):
            return "failed_child_exit"
    if scientific["identity_matches_manifest"] is False:
        return "record_mismatch"
    if scientific["failure"] is not None:
        return "failed_engine"
    if not scientific["record_present"]:
        return "incomplete_observation"
    term = scientific["termination"]
    if record["engine"] == "relu":
        if term["training_censored"] is True or term["diagnostic_censored"] is True:
            return "censored"
        if term["complete"] is True:
            return "planned_horizon_complete"
    else:
        status = scientific["engine_status_recorded"]
        if status == "not_started_deadline":
            return "not_started"
        if status == "failed_without_durable_state":
            return "failed_engine"
        rawterm = term["engine_termination"] or {}
        reason = rawterm.get("reason")
        if reason in ("nonfinite_loss", "exception_durable_partial"):
            return "failed_engine"
        if reason == "loss_stop":
            return "loss_target_stop"
        if reason in ("max_steps", "t_max", "external_stop"):
            return "censored"
    return "outcome_unclassified"


def build_summary(execution_dir, manifest_path=None, expected_arms=28):
    execution_dir = Path(execution_dir).resolve()
    reader = Reader()
    snapshot_path = execution_dir / "MANIFEST.snapshot.json"
    selected = Path(manifest_path).resolve() if manifest_path else snapshot_path if snapshot_path.is_file() else ROOT / "MANIFEST.json"
    manifest = reader.object(selected)
    if manifest is None or not isinstance(manifest.get("configs"), list):
        raise ValueError(f"Manifest unavailable or invalid: {selected}")
    entries = manifest["configs"]
    if expected_arms is not None and len(entries) != expected_arms:
        raise ValueError(f"Expected {expected_arms} declared arms, found {len(entries)}")
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    if len(ids) != len(entries) or len(set(ids)) != len(ids) or any(not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cid) for cid in ids):
        raise ValueError("Manifest arm IDs must be unique safe strings")
    if any(entry.get("engine") not in INDICATORS for entry in entries):
        raise ValueError("Manifest contains an unsupported engine")
    snapshot = reader.object(snapshot_path) if selected != snapshot_path and snapshot_path.is_file() else manifest if selected == snapshot_path else None
    snapshot_matches = canonical(snapshot) == canonical(manifest) if snapshot is not None else None
    if snapshot_matches is False:
        reader.issue(selected, "selected manifest differs from execution snapshot")
    provenance = reader.object(execution_dir / "PROVENANCE.json")
    status = reader.object(execution_dir / "STATUS.json")
    status_source = "STATUS.json" if status else None
    if status is None:
        history = reader.read(execution_dir / "STATE_HISTORY.jsonl", lines=True)
        valid = [row for row in history or [] if isinstance(row, dict)]
        if valid:
            status, status_source = valid[-1], "STATE_HISTORY.jsonl:last_parseable_object"
    status = status or {}
    memberships = {}
    for group in ("pending", "active", "completed"):
        values = status.get(group, [])
        if not isinstance(values, list):
            reader.issue(execution_dir / "STATUS.json", f"{group} must be a list")
            continue
        for value in values:
            cid = value if group == "pending" else value.get("id") if isinstance(value, dict) else None
            if cid not in ids:
                reader.issue(execution_dir, f"unknown launcher arm ID in {group}: {cid!r}")
                continue
            memberships.setdefault(cid, []).append((group, value))
    records = []
    for index, entry in enumerate(entries):
        cid = entry["id"]
        directory = execution_dir / "data" / cid
        membership = memberships.get(cid, [])
        start_issues = len(reader.issues)
        if len(membership) > 1:
            reader.issue(directory, "arm appears multiple times in launcher status")
        group, receipt = membership[0] if len(membership) == 1 else (None, None)
        indicator = directory / INDICATORS[entry["engine"]]
        has_logs = any((execution_dir / "logs" / (cid + suffix)).is_file() for suffix in (".stdout.log", ".stderr.log"))
        if group == "active":
            state = "active" if status.get("status") not in TERMINAL_LAUNCHER else "incomplete"
        elif group == "completed":
            state = "finished"
        elif group == "pending" and not directory.exists() and not has_logs:
            state = "not_started"
        elif directory.exists() or has_logs:
            state = "recorded_output_without_completion_receipt"
        else:
            state = "not_started"
        scientific = (relu_record if entry["engine"] == "relu" else swiglu_record)(directory, entry, reader)
        record = {"id": cid, "manifest_index": index, "engine": entry["engine"],
                  "declared": declared(entry), "output_directory": str(directory),
                  "execution": {"state": state, "launcher_membership": group,
                                "completion_receipt": receipt if group == "completed" else None,
                                "active_receipt": receipt if group == "active" else None,
                                "completion_indicator": indicator.name,
                                "completion_indicator_exists": indicator.is_file(),
                                "output_directory_exists": directory.is_dir(),
                                "log_files_exist": has_logs,
                                "state_note": "Snapshot of recorded files only; process liveness is not probed."},
                  "scientific": scientific, "issues": reader.issues[start_issues:]}
        record["outcome_category"] = classify(record)
        records.append(record)
    categories = Counter(row["outcome_category"] for row in records)
    groups = []
    for engine, teacher in dict.fromkeys((row["engine"], row["declared"]["teacher"]) for row in records):
        chosen = [row for row in records if (row["engine"], row["declared"]["teacher"]) == (engine, teacher)]
        joint = [row["scientific"]["joint_in_own_plateau"]["joint_observed"] for row in chosen]
        groups.append({"engine": engine, "teacher": teacher, "planned_arms": len(chosen),
                       "declared_seeds": [row["declared"]["seed"] for row in chosen],
                       "outcome_counts": dict(sorted(Counter(row["outcome_category"] for row in chosen).items())),
                       "joint_observed_recorded_true": sum(value is True for value in joint),
                       "joint_observed_recorded_false": sum(value is False for value in joint),
                       "joint_observed_unavailable": sum(value is not True and value is not False for value in joint)})
    unexpected_dirs = sorted(p.name for p in (execution_dir / "data").iterdir() if p.is_dir() and p.name not in ids) if (execution_dir / "data").is_dir() else []
    return {
        "schema": "broader-teachers-v2-saved-record-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved JSON bookkeeping and copied diagnostics only; no models, arrays, numerical reevaluation, interpolation, or new event thresholds.",
        "scientific_success": "not_assessed", "expected_arms": expected_arms,
        "declared_arms": len(entries), "reported_arms": len(records),
        "execution_directory": str(execution_dir), "execution_directory_exists": execution_dir.is_dir(),
        "manifest_path": str(selected), "selected_manifest_matches_execution_snapshot": snapshot_matches,
        "manifest_protocol_version": manifest.get("protocol_version"),
        "manifest_runtime": manifest.get("runtime"), "manifest_dependency": manifest.get("dependency"),
        "launcher_status_source": status_source, "launcher_status": status or None,
        "launcher_provenance": provenance, "unexpected_output_directories": unexpected_dirs,
        "outcome_counts": dict(sorted(categories.items())), "groups": groups, "records": records,
        "issues": reader.issues, "source_files_read": list(reader.sources.values()),
        "interpretation": [
            "Every declared arm remains in the denominator, including absent, failed and censored outcomes.",
            "A missing completion indicator after a launcher-recorded child exit is a failed execution even when its exit code is zero.",
            "Completed launcher attempts and complete engine files do not establish convergence or recovery.",
            "SwiGLU update/time caps are censoring; ReLU planned-horizon completion is not a convergence claim.",
            "Persisted ReLU joint fields are copied without recomputing or certifying their legacy heuristic screens.",
            "SwiGLU joint outcomes and explicit prefix censoring remain null unless recorded; null is not false.",
            "Leading and full-rank alignment, architecture-specific plateau rules and refit conventions remain distinct.",
            "Files may change during an active run; this is a snapshot with source hashes, not an atomic engine checkpoint or byte audit of array artifacts.",
        ],
    }


def flat_record(row):
    scientific, execution = row["scientific"], row["execution"]
    term = scientific["termination"]
    swterm = term.get("engine_termination") or {}
    joint = scientific["joint_in_own_plateau"]
    receipt = execution["completion_receipt"] or {}
    return {"id": row["id"], "engine": row["engine"], **row["declared"],
            "outcome": row["outcome_category"], "execution_state": execution["state"],
            "launcher_returncode": receipt.get("returncode"), "launcher_stop_reason": receipt.get("stop_reason"),
            "engine_stop_reason": term.get("training_stop_reason") if row["engine"] == "relu" else swterm.get("reason"),
            "engine_wall_stop_cause": term.get("wall_stop_cause"),
            "observed_updates": term.get("observed_steps") if row["engine"] == "relu" else swterm.get("step"),
            "final_loss": term.get("final_loss") if row["engine"] == "relu" else swterm.get("L"),
            "loss_units": scientific["plateau"]["units"],
            "plateau_last_admissible_update": scientific["plateau"].get("last_admissible_update"),
            "plateau_right_censored": scientific["plateau"].get("plateau_right_censored"),
            "joint_observed_recorded": joint["joint_observed"], "joint_status_recorded": joint["joint_status"],
            "first_joint_saved_step_recorded": joint["first_joint_step"],
            "issue_count": len(row["issues"])}


def markdown(summary):
    lines = ["# Broader-teacher cohort: saved-record summary", "",
             f"All **{summary['reported_arms']}/{summary['declared_arms']} declared arms** are included. Scientific success is not assessed.", "",
             f"Launcher snapshot: `{(summary['launcher_status'] or {}).get('status', 'unavailable')}`. Created: {summary['created_utc']}.", "",
             "Outcome counts: " + "; ".join(f"{key}={value}" for key, value in summary["outcome_counts"].items()) + ".", "",
             "Missing values are shown as `—`; they are not zero or negative events. Loss is raw MSE for ReLU and variance-normalized MSE for SwiGLU. No outcome is interpolated or newly certified.", "",
             "| Arm | Outcome | Observed / planned updates | Engine stop | Own-prefix joint status, as recorded | First joint saved step |",
             "|---|---|---:|---|---|---:|"]
    def shown(value):
        return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")
    for row in summary["records"]:
        flat = flat_record(row)
        lines.append(f"| {shown(flat['id'])} | {shown(flat['outcome'])} | {shown(flat['observed_updates'])} / {shown(flat['planned_updates'])} | {shown(flat['engine_stop_reason'])} | {shown(flat['joint_status_recorded'])} | {shown(flat['first_joint_saved_step_recorded'])} |")
    lines += ["", f"Recorded input/consistency issues: **{len(summary['issues'])}**. See `summary.json` for exact stop reasons, partial records, diagnostic availability, copied checkpoint metrics, launcher receipts, and source hashes.", ""]
    lines += ["- " + note for note in summary["interpretation"]]
    return "\n".join(lines) + "\n"


def write_summary(summary, output_dir, json_name="summary.json"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flats = [flat_record(row) for row in summary["records"]]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(flats[0]) if flats else ["id"])
    writer.writeheader()
    writer.writerows(flats)
    stem = Path(json_name).stem
    contents = {json_name: json.dumps(summary, indent=2, allow_nan=False) + "\n",
                stem + ".csv": buffer.getvalue(), stem + ".md": markdown(summary)}
    for name, value in contents.items():
        path = output_dir / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", "--execution", type=Path, default=ROOT / "execution")
    parser.add_argument("--manifest", type=Path, help="Override manifest; normally prefer the execution snapshot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "latest_summary")
    parser.add_argument("--out", type=Path, help="Exact JSON output path; CSV and Markdown use the same stem beside it")
    parser.add_argument("--expected-arms", type=int, default=28)
    args = parser.parse_args(argv)
    if args.expected_arms < 1:
        parser.error("--expected-arms must be positive")
    summary = build_summary(args.execution_dir, args.manifest, args.expected_arms)
    output_dir = args.out.parent if args.out is not None else args.output_dir
    write_summary(summary, output_dir, args.out.name if args.out is not None else "summary.json")
    print(json.dumps({"declared_arms": summary["declared_arms"], "reported_arms": summary["reported_arms"],
                      "outcome_counts": summary["outcome_counts"], "issues": len(summary["issues"]),
                      "output_directory": str(output_dir.resolve()), "scientific_success": "not_assessed"}, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Summary error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
