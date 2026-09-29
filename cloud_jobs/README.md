# Claude Code sessions as compute workers

Claude Code cloud sessions can't see the Mac, so all of a job's code, configs and instructions live on the `claude-workers` branch. Each session is told which shard to run, runs it, and pushes its results to its own branch. `collect.py` then gathers everything, from any clone, the Mac, or a Claude session with read access.

```
claude-workers branch        worker session (one per shard)          collector
  job folder (reviewed)  ->  fetch branch, pinned env, run shard  ->  collect.py merges all branches
  cloud_jobs/ (this)         push cloud_results/<job>/<shard>/        into one directory
```

## Measured on a Claude cloud container (2026-09-29)

- 2 CPUs and 7 GB RAM per session. Each core is about 2.4× faster than a Lightning Studio core for these engines.
- Reproducibility against the saved Lightning runs:
  - A ReLU rank-2 arm reproduces the saved Lightning run: the same plateau endpoints (65/73), the same first joint checkpoint and the same canonical 1%/5% assessment. Losses differ by at most 3e-15.
  - A SwiGLU rank-16 arm's first 111 updates are bitwise identical to the saved Lightning trajectory.

## Running a job

1. Push the `claude-workers` branch; `push_workers_branch.sh` in the Mac folder does this.
2. For each shard, open a new Claude Code session with the repository `akashkumar-d/compute` attached, and paste that shard's prompt from `jobs/<job>/PROMPTS.md`. Shards are independent: start them in any order and as many at once as you like.
3. Each session sets up the pinned environment, validates the reviewed bundles, runs its arms (one per CPU), keeps itself alive by polling every nine minutes, and pushes to its own branch after each finished arm and at the end.
4. Collect, then analyse the canonical records and small raw files:

   ```sh
   python3 cloud_jobs/collect.py --job <job> --out <dir>
   ```

## Guarantees and limits

- **What runs:** only reviewed arms. `run_shard.py` reuses each bundle's reviewed launcher code to check source, config and manifest hashes, the design gate and the approved review. Engine commands, per-arm budgets and the child environment come from that same code.
- **Scheduler:** the only new code is a small process pool sized to the container.
- **Snapshots:** large parameter snapshots stay in the session and are lost when it ends. Their hashes are kept, and the canonical records are computed while the snapshots still exist. A rerun is deterministic if a snapshot is ever needed.
- **Failure handling:** a session that dies loses its unfinished arms, but finished arms are already pushed. Rerun a missing arm in a fresh shard under a new shard id; never overwrite results.
- **Session lifetime:** sessions stay alive only while they poll. The protocol polls every 9 minutes; shards are sized to about 70 minutes or less.
