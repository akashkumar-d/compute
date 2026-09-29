#!/usr/bin/env python3
"""Write SHARDS.json and PROMPTS.md for the ten-seed high-rank cohort.

Run from the repository root:  python3 cloud_jobs/jobs/highrank_s10_20260929/make_shards.py
Every preregistered arm of the three reviewed bundles appears in exactly one shard.
Shards are sized for a 2-CPU Claude Code container (~70 minutes or less each).
"""
import hashlib
import json
from pathlib import Path

JOB = 'highrank_s10_20260929'
LEAF = 'population_agop_highrank_seeds10_20260929'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SUMMARIZER = 'population_agop_rank_sweep_20260928/analysis/summarize_coverage.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    bundles, cell_arms = {}, {}
    for b in ('a01', 'a02', 'a03'):
        root = REPO / LEAF / f'bundle_{b}'
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        bundles[b] = dict(path=f'bundle_{b}', manifest_sha256=sha(root / 'MANIFEST.json'),
                          execution_dir=f'execution_{b}_cloud', arms=len(manifest['configs']))
        for e in manifest['configs']:
            cell_arms.setdefault(e['cell'], []).append((b, e['id'], e['seed']))
    for v in cell_arms.values():
        v.sort(key=lambda t: t[2])
    get = lambda cell: [[b, a] for b, a, _ in cell_arms[cell]]
    shards = {}
    shards['R01'] = dict(arms=sum((get(c) for c in (
        'v11_hr_relu_h2_r16', 'v11_hr_relu_h2_r8', 'v11_hr_relu_gaussian_rbf_r8', 'v11_hr_relu_abs_r8',
        'v11_hr_relu_h2_r4', 'v11_hr_relu_h2_r2')), []), expected_minutes=40,
        note='All 56 ReLU arms (~70-200 s each on a 2-CPU container).')
    long_arms = sum((get(c) for c in ('v11_hr_swiglu_h2_r16', 'v11_hr_swiglu_h2_headslow_r16',
                                      'v11_hr_swiglu_h2_r8', 'v11_hr_swiglu_h2_headslow_r8')), [])
    for i in range(0, len(long_arms), 4):
        shards[f'S{i // 4 + 1:02d}'] = dict(arms=long_arms[i:i + 4], expected_minutes=70,
                                            note='Four SwiGLU r8/r16 arms, up to 1990 s each (two waves).')
    mid = sum((get(c) for c in ('v11_hr_swiglu_abs_r8', 'v11_hr_swiglu_gaussian_rbf_r8')), [])
    for i in range(0, len(mid), 4):
        shards[f'S{11 + i // 4:02d}'] = dict(arms=mid[i:i + 4], expected_minutes=70,
                                             note='Four SwiGLU abs/RBF r8 arms, up to 1990 s each.')
    r4, r2 = get('v11_hr_swiglu_h2_r4'), get('v11_hr_swiglu_h2_r2')
    shards['S16'] = dict(arms=r4[:5] + r2[:5], expected_minutes=45, note='SwiGLU h2 r4/r2, seeds 9351-9355.')
    shards['S17'] = dict(arms=r4[5:] + r2[5:], expected_minutes=45, note='SwiGLU h2 r4/r2, seeds 9356-9360.')
    listed = [a for s in shards.values() for _, a in s['arms']]
    total = sum(v['arms'] for v in bundles.values())
    assert len(listed) == len(set(listed)) == total == 136, (len(listed), total)
    spec = dict(job=JOB, leaf=LEAF, bundles=bundles, global_seconds=5400,
                summarizer=dict(path=SUMMARIZER, sha256=sha(REPO / SUMMARIZER)),
                pack_exclude=['states/*', 'resume.npz', '*_snaps.npz'],
                engine_environment='cloud_jobs/worker/setup_env.sh (CPython 3.12, numpy 1.26.4, scipy 1.11.4, clarabel 0.11.1)',
                shards=shards)
    (HERE / 'SHARDS.json').write_text(json.dumps(spec, indent=1) + '\n')
    lines = [f'# Prompts for job `{JOB}`', '',
             'Start one Claude Code session per shard with the repository `akashkumar-d/compute` attached, '
             'and paste its prompt. Shards are independent; start them in any order and as many at once as '
             'you like. Each finishes in about the listed time on a 2-CPU session and pushes its results to '
             "that session's own branch.", '',
             '| Shard | Arms | Expected | Contents |', '|---|---:|---:|---|']
    for sid, s in shards.items():
        lines.append(f"| {sid} | {len(s['arms'])} | ~{s['expected_minutes']} min | {s['note']} |")
    for sid in shards:
        lines += ['', f'## {sid}', '', '```text',
                  f'Compute worker for job {JOB}, shard {sid}. In this repository run: '
                  'git fetch origin claude-workers && git worktree add /tmp/jobs origin/claude-workers . '
                  f'Then follow /tmp/jobs/cloud_jobs/WORKER.md exactly with JOB={JOB} and SHARD={sid}. '
                  'Do not change any code or configuration. Keep this session alive by polling until the '
                  "shard finishes, push the results to this session's branch, and reply with the branch name "
                  'and the final status.', '```']
    (HERE / 'PROMPTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: len(v['arms']) for k, v in shards.items()}))


if __name__ == '__main__':
    main()
