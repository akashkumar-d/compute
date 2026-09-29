# Prompts for job `highrank_s10_20260929`

Start one Claude Code session per shard with the repository `akashkumar-d/compute` attached, and paste its prompt. Shards are independent; start them in any order and as many at once as you like. Each finishes in about the listed time on a 2-CPU session and pushes its results to that session's own branch.

| Shard | Arms | Expected | Contents |
|---|---:|---:|---|
| R01 | 56 | ~40 min | All 56 ReLU arms (~70-200 s each on a 2-CPU container). |
| S01 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S02 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S03 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S04 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S05 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S06 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S07 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S08 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S09 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S10 | 4 | ~70 min | Four SwiGLU r8/r16 arms, up to 1990 s each (two waves). |
| S11 | 4 | ~70 min | Four SwiGLU abs/RBF r8 arms, up to 1990 s each. |
| S12 | 4 | ~70 min | Four SwiGLU abs/RBF r8 arms, up to 1990 s each. |
| S13 | 4 | ~70 min | Four SwiGLU abs/RBF r8 arms, up to 1990 s each. |
| S14 | 4 | ~70 min | Four SwiGLU abs/RBF r8 arms, up to 1990 s each. |
| S15 | 4 | ~70 min | Four SwiGLU abs/RBF r8 arms, up to 1990 s each. |
| S16 | 10 | ~45 min | SwiGLU h2 r4/r2, seeds 9351-9355. |
| S17 | 10 | ~45 min | SwiGLU h2 r4/r2, seeds 9356-9360. |
| R01b | 20 | ~12 min | Declared rerun of 20 ReLU h2 r4/r2 arms from R01 (environment modified during R01). |

R01 and R01b already ran in the setup session (R01b is a declared rerun of 20 R01 arms). Start sessions only for S01-S17.

## R01

```text
Compute worker for job highrank_s10_20260929, shard R01. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=R01. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S01

```text
Compute worker for job highrank_s10_20260929, shard S01. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S01. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S02

```text
Compute worker for job highrank_s10_20260929, shard S02. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S02. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S03

```text
Compute worker for job highrank_s10_20260929, shard S03. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S03. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S04

```text
Compute worker for job highrank_s10_20260929, shard S04. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S04. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S05

```text
Compute worker for job highrank_s10_20260929, shard S05. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S05. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S06

```text
Compute worker for job highrank_s10_20260929, shard S06. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S06. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S07

```text
Compute worker for job highrank_s10_20260929, shard S07. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S07. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S08

```text
Compute worker for job highrank_s10_20260929, shard S08. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S08. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S09

```text
Compute worker for job highrank_s10_20260929, shard S09. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S09. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S10

```text
Compute worker for job highrank_s10_20260929, shard S10. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S10. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S11

```text
Compute worker for job highrank_s10_20260929, shard S11. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S11. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S12

```text
Compute worker for job highrank_s10_20260929, shard S12. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S12. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S13

```text
Compute worker for job highrank_s10_20260929, shard S13. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S13. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S14

```text
Compute worker for job highrank_s10_20260929, shard S14. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S14. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S15

```text
Compute worker for job highrank_s10_20260929, shard S15. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S15. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S16

```text
Compute worker for job highrank_s10_20260929, shard S16. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S16. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## S17

```text
Compute worker for job highrank_s10_20260929, shard S17. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=S17. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```

## R01b

```text
Compute worker for job highrank_s10_20260929, shard R01b. In this repository run: git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB=highrank_s10_20260929 and SHARD=R01b. Do not change any code or configuration. Keep this session alive by polling until the shard finishes, push the results to this session's branch, and reply with the branch name and the final status.
```
