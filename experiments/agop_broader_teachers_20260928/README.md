# Higher-rank broader-teacher development batch

Prepared in response to the request to reconsider higher-rank training and include more teacher links. Read PROTOCOL.md for the fixed 28-arm design and limits. No outcome has been assumed.

Use only dispatch_server.sh (Linux, AGOP_EXECUTION_SITE=SERVER) for remote execution, or launcher/launch.py --dry-run for static validation. The copied legacy engine scripts are not approved direct local entrypoints. No training is performed on the Mac. NumPy/SciPy come from the remote environment; a batch-private virtual environment adds the pinned Clarabel solver without modifying the shared environment. All actual versions are saved.

The remote run waits for the named existing CPU job before up to55minutes of four-worker compute. A two-hour queue timeout leaves all runs unstarted. Every configuration, failed/partial result and warning remains in the record. Development: two seeds per cell, not a confirmation experiment.

The current arXiv and ICLR files are untouched. DISPATCH.json and CHECKPOINT.md will record the exact remote revision, command, status and restart/pull instructions after dispatch. Source hashes are finalized only after review.
