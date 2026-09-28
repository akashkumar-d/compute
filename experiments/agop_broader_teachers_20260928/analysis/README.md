# Saved-record cohort summary

This standard-library-only tool reads the frozen manifest, launcher receipts, and named JSON outputs. It imports no scientific engine and loads no NumPy arrays. Every declared arm remains in manifest order, including unstarted, failed, capped, and partially diagnosed attempts. Missing metrics remain JSON `null` (blank CSV cells), never zero or false.

From the `broader_teachers_v2` directory, the dispatch command is exactly:

```sh
python3 -B analysis/summarize.py --execution execution --out execution/summary.json
```

This writes `execution/summary.json`, `execution/summary.csv`, and `execution/summary.md`. Rerunning replaces only those generated summaries. It does not modify engine outputs or the manifest. The JSON carries input-file byte hashes, provenance, all 28 records, diagnostic availability, source inconsistencies, exact engine/launcher stop reasons, and copied checkpoint metrics. The Markdown table is a compact execution overview; use the JSON for scientific details. The default expected manifest size is 28; a mismatch is an error.

For a locally pulled execution directory, from the same bundle directory:

```sh
python3 -B analysis/summarize.py --execution /absolute/path/to/pulled/execution --out analysis/local_summary/summary.json
```

The execution's `MANIFEST.snapshot.json` is preferred. If absent, the bundle's `MANIFEST.json` is used. `--manifest /absolute/path/MANIFEST.json` explicitly overrides this and records any mismatch with the execution snapshot. Launcher status uses `STATUS.json`, falling back to the last parseable object in `STATE_HISTORY.jsonl`. No process-liveness claim is made from an old receipt.

A launcher-recorded child exit without `DONE.json` (ReLU) or `result.json` (SwiGLU) is classified as a failed execution even when its exit code is zero. Engine caps/censoring remain distinct from execution failure and planned-horizon completion. A saved SwiGLU `completed=true` flag is not convergence.

ReLU's persisted `joint_observed`, `joint_status`, `first_joint_step`, and related fields are copied directly. Its first qualifying saved checkpoint is included when present. An observed event can coexist with later censoring. Heuristic spectral/refit warnings are preserved; no certificate is issued. The current SwiGLU driver persists no joint-event classification or explicit prefix-censoring flag, so those fields remain unavailable. The summarizer does not compute new thresholds, infer joint events from separate maxima, interpolate crossings, or recompute prefixes. Additional before/after-prefix threshold scanning is deferred to a separately reviewed saved-record analysis.

Run the tiny synthetic bookkeeping checks locally or on the server:

```sh
python3 -B -m unittest discover -s analysis -p 'test_summarize.py' -v
```

These checks write only temporary synthetic JSON fixtures and evaluate no model or population moments.
