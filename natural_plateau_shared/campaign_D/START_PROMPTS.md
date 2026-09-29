# Campaign D: the eight session prompts

**Once, on the Mac:** run `bash push_A_files.sh` in `natural_plateau_shared/`. This puts the campaign files on GitHub.

**Then start eight Claude Code sessions** (web or app), four on each account:

| account | worker IDs |
|---|---|
| Account 1 | S1, S2, S3, S4 |
| Account 2 | S5, S6, S7, S8 |

For each session:

1. Choose the repository `akashkumar-d/compute`.
2. Choose its default branch, `claude/focused-gauss-cy16t3`.
3. Paste the prompt below, changing only the ID on its first line.

**About the IDs:**

- Each ID owns a different share of the work; every share starts with its longest p = 97 runs.
- An ID that is never started: the other sessions take over its runs.
- An ID started twice: the two sessions split its runs.

---

## Prompt (change only the first line)

```
Your worker ID is S1.

You are a worker of campaign D, a batch of numpy CPU training runs for a research project (natural loss plateaus in
modular arithmetic, across moduli p). Everything is already written and tested. Your only job is to run it with your
worker ID and keep this session active until the campaign is finished (about 8-10 hours).

Work in this repository and stay on the git branch this session was set up with.
1. If natural_plateau_shared/campaign_D/worker.py does not exist, run: git pull origin claude/focused-gauss-cy16t3
2. Read natural_plateau_shared/campaign_D/WORKER.md and follow it exactly. Wherever it says SX, use your worker ID.

In short: install numpy if it is missing; run the selftest (Bash timeout 600000 ms); then run the watch command (Bash
timeout 600000 ms) again and again until it prints ALL DONE. Do not edit, commit or delete any file yourself: the worker commits and pushes its
own results. The repository is public: never write tokens or passwords anywhere.
```

**First lines, ready to copy:**

- Account 1: `Your worker ID is S1.` · `Your worker ID is S2.` · `Your worker ID is S3.` · `Your worker ID is S4.`
- Account 2: `Your worker ID is S5.` · `Your worker ID is S6.` · `Your worker ID is S7.` · `Your worker ID is S8.`

## Afterwards

- **Progress.** Ask `claude/plateau-review` (the Cowork session), or run
  `python3 natural_plateau_shared/campaign_D/worker.py status --all` in any session.
- **A session stops early.** Start a new one with the same ID and the same prompt. Its finished runs are kept; other sessions may
  already have taken over part of its unfinished runs. If several sessions stopped (in campaign C most lasted about 8 hours),
  start them all again the same way: finished runs are never repeated.
- **A session says `STOP: chains keep failing on this machine`.** Its runs move to the other sessions. Tell
  `claude/plateau-review`.
- **When the sessions print ALL DONE.** They print it only when every run is finished. `claude/plateau-review` then collects the
  results from every session branch and writes the analysis.
- **Repository.** Keep it readable until then. Make it private afterwards.
