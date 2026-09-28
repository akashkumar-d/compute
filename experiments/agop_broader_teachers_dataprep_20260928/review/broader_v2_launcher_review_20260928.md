# Broader-teacher v2 launcher: independent implementation review

**Verdict: PASS for the reviewed guarded launcher/dispatch implementation, after two requested fixes.** A source-hash freeze is still required before dispatch; this is not authorization to bypass preflight or proof that a real server batch has completed safely.

## Scope and validation

Only standard-library mocks and static source inspection were used. No scientific engine was imported or executed, no local training occurred, no network call was made, and no actual child process or process-group signal was issued by the tests. The author's files were copied into this review workstream before running tests, so test fixtures did not modify the bundle.

**21 tests passed:** 15 author tests plus 6 independent checks. `bash -n` also passed for `dispatch_server.sh`.

## Requested fixes, now resolved

1. The first revision accepted a manifest with more than four workers or more than 480 seconds per arm. Validation now enforces workers in 1–4, per-arm <=480, and total <=3300 seconds. Independent rejection tests pass.
2. The first cleanup requested SIGKILL and immediately removed the active record without acknowledging process exit. The revision starts hard termination before the deadline and performs bounded `wait`/reaping. An OS-unreapable child remains explicitly active with a null return code, `reap_acknowledged=false`, and `cleanup_status=kill_requested_unreaped`; it is not falsely reported as cleaned up. Both ordinary reaping and the unkillable mock pass.

## Reviewed contracts

- **Server boundary:** `dispatch_server.sh` rejects non-Linux or missing `AGOP_EXECUTION_SITE=SERVER` before environment setup/science imports. The launcher itself imports only standard-library modules and calls its Linux+SERVER guard before output creation/child launch. Server preflight likewise guards its numerical imports. The legacy ReLU runner imports scientific libraries before its own weaker direct-run guard; it is explicitly **not an approved standalone entrypoint**. This review approves dispatch/launcher entrypoints, preserving the numerical kernels.
- **Fresh output:** safe single-component execution names, existing-output refusal, exclusive log creation, preserved partial data and complete pending/completed/active identities. No resume/truncation of earlier attempts.
- **Manifest:** nonempty source hashes, exact declared config content, safe in-bundle resolved file paths, supported engine names, duplicate detection, initial manifest digest retention, and revalidation while queued/before every child start. At this review's snapshot, the actual staged manifest still has zero source hashes and is correctly refused by both validator and preflight; root must freeze and statically validate it before remote use.
- **Queue:** only `modarith-p61-decay-grid-20260928-v2` is inspected; readiness needs an integer exit record and a non-live recorded PID. Zombies count as nonexecuting. PID/exit records are reread for stability, and readiness is rechecked before each child launch. No signal is sent to that dependency. Polling is 30 seconds, waiting is separately capped at the manifest's 7200 seconds, and queue time does not consume the 3300-second compute clock. Dependency readiness does not establish exclusive server access.
- **Resources:** four single-thread workers, external parent-enforced per-arm/global wall limits, process groups created by `start_new_session=True`, and termination restricted to recorded children of this invocation. Ordinary exits also clear surviving members of their own group. Signal/error paths enter cleanup. Partial files stay available even when a completion indicator is missing; unstarted arms remain listed.
- **Engine arguments:** at a full 480-second allocation, ReLU receives `--max-seconds 470 --diagnostic-reserve 140`; SwiGLU receives `--wall-seconds 330 --diagnostic-seconds 140`. Both receive an absolute soft deadline, correct config arguments, distinct fresh output paths, and the SwiGLU tag. These match the actual runner parsers. A shortened ReLU arm preserves the existing runner's `min(reserve, 0.4*soft_budget)` clamp, so 140 seconds is the normal full-budget reserve, not a guarantee for shortened tail allocations.
- **Dispatcher:** private `runtime` virtual environment (inheriting server NumPy/SciPy), pinned Clarabel requirement, version receipt, bounded server preflight (95-second timeout plus 5-second kill grace), then launcher and standard-library summary. No hardware change or GPU request. Provisioning and queue waiting are separate from the 55-minute batch compute ceiling; total end-to-end wall time must not be described as <=55 minutes.

## Limits

Mock tests cover control flow and subprocess contracts, not actual Linux signal latency, operating-system stalls, scientific kernel correctness, or runtime throughput. A process stuck in an uninterruptible OS state cannot be guaranteed dead by a bounded user-space wait; the final receipt now accurately exposes that unresolved state for follow-up. No absolute no-orphan guarantee is claimed in that exceptional case. No scientific-success conclusion follows from a zero launcher exit code alone.

## Reviewed hashes

- `launcher/launch.py`: `7667508a90bb5905f7bfc52c0e98f6d66eb4a3ed686a8cb942eb83b0a2fd6244`
- `launcher/test_launch.py`: `ca96f1ec63a63a31f24a5925c3b4c450e4788c09de167af94870c240f0e6e1e3`
- `dispatch_server.sh`: `8407724b9365f7711de3c81247a16bb679fbd35732ac13fa81b0f86ec92ad0ee`
- `code/scaling_run.py`: `e941d0b71dbf11fa56257b71b1aeea7bd4aa13281328c0b4a5891436f4d424a7`
- `swiglu/code/run_one.py`: `0b4c3739548c544142945333eb5101d659fd2ac2c8e9727674e627481f21bb6e`

Reproduction: copied author test module and launcher plus `test_independent_review.py` are under `review/broader_v2_launcher_mock_review/`. Run only the standard-library unittest suite there. No canonical or launcher file was edited by this reviewer.
