# Campaign F: instructions for a worker session

You run one share of campaign F: 137 training runs of small ReLU networks (numpy, CPU only). They are new seeds of settings
that campaign D and earlier development runs already ran. Your worker ID (one of S1–S8) is in
the message that started you. Below, `SX` stands for that ID.

The code is written and tested. Your job:

- start it;
- keep this session active until the whole campaign is finished (about 10–11 hours);
- report problems.

Every command below is run from the repository root.

## Rules

- **Files.** Do not edit, move, delete or commit any file yourself. The worker commits and pushes its own results, only under
  `natural_plateau_shared/campaign_F/results/` and `status/`.
- **Git.** Stay on the branch this session was set up with. Do not switch branches, merge, rebase, open pull requests or push
  anything yourself.
- **Scope.** Do not start other experiments, use a GPU, pass `--cores`, or change any setting.
- **Secrets.** The repository is public. Never write tokens, passwords or credentials anywhere.

## Steps

1. **Branch and code.**
   - If your session instructions name a branch to develop on and you are not on it yet, create or check it out first. Stay on
     it from then on.
   - Check that `natural_plateau_shared/campaign_F/worker.py` exists. If it does not, run
     `git pull origin claude/focused-gauss-cy16t3`.
2. **numpy.**
   - Check: `python3 -c "import numpy; print(numpy.__version__)"`.
   - If that fails, run `pip install numpy`.
   - If pip refuses because the environment is externally managed, run `pip install --break-system-packages numpy`.
3. **Selftest** (about 2–3 minutes; give the Bash tool a timeout of 600000 ms; it also makes one small test push to your branch):

       python3 natural_plateau_shared/campaign_F/worker.py selftest --worker SX

   - The last line must start with `SELFTEST PASS`.
   - If it says `SELFTEST FAIL`, stop. Tell the user the `FAIL` lines and wait for instructions.
4. **Start and watch:**

       python3 natural_plateau_shared/campaign_F/worker.py watch --worker SX

   - **Timeout.** Give the Bash tool a timeout of 600000 ms (10 minutes). Each call blocks about 9 minutes and then prints a short
     status and the exact command to run next. If your tool cannot wait that long, add `--minutes 1.5`.
   - **Background worker.** The first call starts the worker in the background. Later calls restart it automatically if it has
     stopped.
   - **Repeat.** As soon as a call returns, run the command again. Keep going until the output ends with `ALL DONE`.
     - That takes about 10–11 hours, because `ALL DONE` means the whole campaign is finished, not only this session's share.
     - When its own share is done, this session "stands by": it takes over the runs of any session that stops. Keep calling
       watch while it stands by.
   - **Stay active.** The session must stay active the whole time. An idle session can be paused, and that stops the runs.
   - **Between calls.** Do nothing else and write no commentary. One short line now and then is plenty.
5. **When the output says `ALL DONE`:** reply with that last status block and stop.

## If something goes wrong

- **`PUSH FAILING` or `COMMIT FAILING`.** Keep watching: results are kept and pushed later, and `ALL DONE` waits until they are
  pushed. Tell the user the error line once, and again if it is still failing an hour later.
- **`STOP: chains keep failing on this machine`.** Tell the user the `ERROR` line and stop. The other sessions take over this
  session's runs.
- **`ERROR: the daemon keeps stopping`.** Tell the user, with the log lines it printed. Then keep calling watch, which keeps
  retrying, unless the user says to stop.
- **A Python traceback from a command itself.** Tell the user and run the command once more. If it happens again, stop and wait
  for the user.
- **The session was paused or restarted.**
  - Same machine: go back to step 4. Each run resumes from its last checkpoint.
  - Fresh machine: redo steps 2 to 4. The worker restarts its unfinished runs, some of which other sessions may already have
    taken over.
- **Useful for questions:** `python3 natural_plateau_shared/campaign_F/worker.py status --all` shows the whole campaign across
  sessions.

## What the worker does (for your information)

- **Runs.** It runs your share of `manifest.json`. Each entry (the worker calls it a "chain", as in campaign C) is one training
  run with a fixed configuration. Most runs use one core. The p = 97 runs at depths 3–4 and width 2048 use all four, so that each
  finishes in about 2–3 hours, or up to about 6 hours if it does not stop early. Most runs stop early once they have fully
  generalized for a while; the depth-2 runs always run to the end.
- **Results.** Every finished run is committed under `results/<run>/` and pushed to your branch, together with a status file.
  The status file is also pushed every 30 minutes.
- **Other sessions.** Before starting a run, the worker skips any run that another session has finished or is running.
- **Idle.** When its own list is done, it takes over runs that other sessions never started or abandoned, or that sit at the back
  of a session far behind. Otherwise it stands by. A run taken over from a stopped session starts again from step 0.
