# natural_plateau_shared: working across two Claude accounts

This folder is how two Claude sessions on different accounts share the natural-plateau experiments without getting in each other's way. Both sessions read this file first, every time.

**Where it lives.** The folder is `natural_plateau_shared/` in the GitHub repo `akashkumar-d/compute`, branch `claude/focused-gauss-cy16t3`.
- The repo copy is the only shared copy.
- Each account works in its own clone and syncs through git.
- The repo is **public**: never commit tokens, passwords or links containing `?token=`.
- **How A's changes get there.** A's Cowork session can read the repo but cannot push to it. A writes its files in the copy of this folder on the user's Mac. The user then runs `bash push_A_files.sh` there, which:
  - pulls first;
  - copies only A-owned files, so it never touches B's log, claims or results;
  - commits and pushes.
- B pushes directly from its own session.

## Who is who

| ID | What it is | Main job |
|---|---|---|
| **A** = `claude/plateau-review` | Cowork, main account | Designs the tasks, analyses results, writes the report. Owns `README.md` and `TASKS.md`. Also mirrors milestones into the lab hub on the user's computer. |
| **B** = `claude/plateau-runner` | Claude Code cloud session, second account | Runs the tasks in `TASKS.md` and reports back. |

The full research write-up lives on the user's computer, not in this repo. Ask the user if you need it. `TASKS.md` gives enough context to run every task.

## Folder layout

- `README.md`: this protocol. Owned by A.
- `TASKS.md`: task list, with IDs, batch files and status. Owned by A.
- `code/natplat_server.py`: the learner and runner. It is shared and **read-only**.
- `code/tasks/<TASK>.json`: exact configs for each task. Owned by A.
- `claims/`: one empty file per claimed task, named `<TASK>.<A|B>`.
- `log/A.md` and `log/B.md`: append-only progress logs, one per account.
- `inbox/`: messages between A and B, one new file per message.
- `results/<TASK>/`: `*.json.gz` runs, `SUMMARY.md` and `queue.log`, written by whoever claimed the task.

## Git rules

1. `git pull --rebase` before reading anything and before every commit.
2. Commit small and push right away. Never force-push, and never rewrite or delete someone else's commits.
3. **Never commit:**
   - checkpoints (`*.ckpt.npz`), which `.gitignore` excludes;
   - any file over 50 MB;
   - scratch files;
   - credentials of any kind.
4. **If a push is rejected:** run `git pull --rebase`, re-check `claims/`, and push again.
5. **Commit messages:** `<A|B> <TASK>: <what>`, for example `B T2: claim` or `B T2: results s0-s2 + SUMMARY`.

## Rules

1. **Start of every session:**
   - `git pull --rebase`.
   - Read this file, `TASKS.md`, the last 20 lines of `log/A.md` and `log/B.md`, `claims/`, and any `inbox/` files addressed to you that are newer than your last log entry.
2. **One writer per file.**
   - Each account writes only:
     - its own log;
     - its own claim files;
     - `results/<TASK>/` for tasks it has claimed;
     - new files in `inbox/`.
   - Never edit or delete another account's files.
   - To change `TASKS.md`, `README.md` or the code, B sends a message to A through `inbox/`.
3. **Claim before running.**
   - Check `claims/`. If `<TASK>.A` or `<TASK>.B` already exists, the task is taken.
   - Otherwise create an empty `claims/<TASK>.<you>`, add a `started` line to your log, then commit and push.
   - Only start running once that push has succeeded. If it was rejected and someone else claimed the task meanwhile, back off.
4. **Run with the shared code, unchanged.** From inside `natural_plateau_shared/`:

       python3 code/natplat_server.py run code/tasks/<TASK>.json results/<TASK> --workers N

   - Use N ≤ the number of free CPU cores, and at most one worker per run.
   - The runner saves a local checkpoint every 5,000 steps. It writes `results/<TASK>/<run>.json.gz` when a run finishes and deletes that run's checkpoint.
   - Rerunning the same command resumes unfinished runs and skips finished ones. The numbers come out identical.
   - Never edit the learner. If something needs changing, propose it in `inbox/` first.
5. **Cloud sessions pause when idle.**
   - Background processes can die when a chat turn ends and the session goes quiet, so keep the session active while runs execute. For example, check `python3 code/natplat_server.py status results/<TASK>` every ~10 minutes.
   - After an interruption, rerun the same command. Checkpoints stay in the container but are not committed.
   - Commit and push each finished `*.json.gz` as soon as it appears, so a finished run is never lost.
6. **Report when a task finishes:**
   - Run `python3 code/natplat_server.py summary results/<TASK> > results/<TASK>/SUMMARY.md`.
   - Add one line to your log: `UTC time | <you> | <TASK> | done | 2–3 key numbers | results/<TASK>/SUMMARY.md`.
   - Commit and push.
   - A copies milestone lines into the lab hub; B does not need to.
7. **Stay in scope:**
   - Do not start runs beyond what `TASKS.md` lists: no extra seeds, sweeps or GPU jobs without asking the user.
   - Keep only `*.json.gz`, `SUMMARY.md` and `queue.log` in `results/`.
8. **Never commit credentials.** No tokens, passwords or tokenized notebook URLs anywhere; the repo is public. If a server needs a login, the user logs in.

## How A checks on B

A runs `git pull --rebase`, then reads:
- `log/B.md`
- `claims/`
- `results/*/SUMMARY.md`
- `inbox/`

A then updates the Status column of `TASKS.md` (commit and push) and its report on the user's computer.

## Where the numbers come from

Every run is fully determined by its config: fixed data split and fixed initialization seed. Checkpoint resumes give identical numbers. So a run done by A and one done by B with the same config are interchangeable.
