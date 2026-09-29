# Compute-worker protocol for a Claude Code cloud session

Your prompt names a JOB and a SHARD. You are a compute worker: you run that one preregistered shard of an already reviewed job, keep this session alive until it finishes, and push the results. You do not write, change or interpret code, configurations or science.

## Rules

- Do not edit, delete, move or reformat any existing file. The only files you add are under `cloud_results/<JOB>/<SHARD>/` in this session's repository, and the scripts below write them.
- Push only to this session's own branch (`git branch --show-current`). Never push to `claude-workers`, any `codex/*` branch or the default branch.
- Run only your SHARD, once. Never rerun a shard whose results already exist. Never change seeds, configs, time budgets, thresholds or the environment versions.
- If a step fails, do not improvise a workaround. Push whatever exists (step 6) and report the exact error text.

## Steps

Replace JOB and SHARD in every command.

1. **Jobs checkout** (read-only; the prompt usually did this already):

   ```sh
   git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers
   ```

2. **Pinned environment** (about a minute; installs Python 3.12 with numpy 1.26.4, scipy 1.11.4 and clarabel 0.11.1):

   ```sh
   bash /tmp/jobs/cloud_jobs/worker/setup_env.sh
   ```

3. **Plan check** (validates the reviewed bundles and prints the arms, workers and budgets; runs nothing):

   ```sh
   /tmp/agopenv/bin/python /tmp/jobs/cloud_jobs/worker/run_shard.py --job JOB --shard SHARD --jobs-root /tmp/jobs --results-root "$(git rev-parse --show-toplevel)/cloud_results" --dry-run
   ```

4. **Start the shard in the background**, exactly once:

   ```sh
   nohup /tmp/agopenv/bin/python /tmp/jobs/cloud_jobs/worker/run_shard.py --job JOB --shard SHARD --jobs-root /tmp/jobs --results-root "$(git rev-parse --show-toplevel)/cloud_results" > /tmp/shard_SHARD.log 2>&1 &
   ```

5. **Keep the session alive.** Repeat the following as one command per tool call, within the 10-minute tool timeout:

   ```sh
   sleep 540; bash /tmp/jobs/cloud_jobs/worker/poll.sh JOB SHARD
   ```

   Each poll prints progress and pushes any newly finished arms. Continue until the state is `finished`, `interrupted` or `global_budget_exhausted`, or the poll says `runner process: not running`.

6. **Finish:**

   ```sh
   bash /tmp/jobs/cloud_jobs/worker/sync_results.sh JOB SHARD final
   ```

   Then reply with the branch name, the final state, arms done/total, and any arm with a nonzero return code or a missing completion indicator.

## What gets pushed

- `arms/<arm>.tar.gz`: the arm's small result files and logs. Large parameter snapshots are not committed.
- `arms/<arm>.inventory.json`: sha256 and size of every file, including the snapshots that were left out.
- `canonical/<bundle>.json`: canonical per-arm records, computed in this session while the snapshots still exist.
- `SHARD_STATUS.json` and `ENVIRONMENT.json`: progress, return codes, versions, CPU and the jobs commit.
