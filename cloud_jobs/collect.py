#!/usr/bin/env python3
"""Collect pushed shard results for one job from every remote branch.

Usage (inside any clone of akashkumar-d/compute; needs only read access):
    python3 cloud_jobs/collect.py --job highrank_s10_20260929 --out <new_or_existing_dir>

For each shard in SHARDS.json (read from origin/claude-workers) it finds every remote
branch holding cloud_results/<job>/<shard>/SHARD_STATUS.json, keeps the one with the
most finished arms (ties: newest commit), extracts it, unpacks each arm into
<out>/data/<arm>/, copies canonical per-arm records to <out>/canonical/, and writes
<out>/COLLECTION.json with per-shard branch/commit, done/missing arms and failures.
Read-only with respect to the repository (fetch only). Standard library only.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile


def git(*args, binary=False):
    out = subprocess.run(['git', *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs-ref', default='origin/claude-workers')
    ap.add_argument('--no-fetch', action='store_true')
    a = ap.parse_args()
    if not a.no_fetch:
        git('fetch', '-q', 'origin', '+refs/heads/*:refs/remotes/origin/*')
    spec = json.loads(git('show', f'{a.jobs_ref}:cloud_jobs/jobs/{a.job}/SHARDS.json'))
    refs = [r.strip() for r in git('for-each-ref', '--format=%(refname)', 'refs/remotes/origin').splitlines()
            if not r.endswith('/HEAD')]
    found = {}
    for ref in refs:
        names = git('ls-tree', '-r', '--name-only', ref, f'cloud_results/{a.job}/').splitlines()
        for shard in {n.split('/')[2] for n in names if n.endswith('/SHARD_STATUS.json')}:
            status = json.loads(git('show', f'{ref}:cloud_results/{a.job}/{shard}/SHARD_STATUS.json'))
            when = int(git('log', '-1', '--format=%ct', ref).strip())
            key = (status.get('arms_done', 0), status.get('state') == 'finished', when)
            if shard not in found or key > found[shard]['key']:
                found[shard] = dict(ref=ref, commit=git('rev-parse', ref).strip(), key=key, status=status)
    out = a.out
    (out / 'data').mkdir(parents=True, exist_ok=True)
    (out / 'canonical').mkdir(exist_ok=True)
    (out / 'status').mkdir(exist_ok=True)
    report = dict(job=a.job, jobs_ref=a.jobs_ref, jobs_commit=git('rev-parse', a.jobs_ref).strip(), shards={})
    for shard, s in spec['shards'].items():
        expected = [arm for _, arm in s['arms']]
        if shard not in found:
            report['shards'][shard] = dict(state='not_found', done=[], missing=expected)
            continue
        f = found[shard]
        prefix = f'cloud_results/{a.job}/{shard}'
        archive = git('archive', '--format=tar', f['commit'], prefix, binary=True)
        done, failed = [], []
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                rel = m.name[len(prefix) + 1:]
                data = tar.extractfile(m).read()
                if rel.startswith('arms/') and rel.endswith('.tar.gz'):
                    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as inner:
                        for im in inner.getmembers():
                            if im.isfile() and not im.name.startswith(('/', '..')) and '..' not in im.name:
                                target = out / im.name
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(inner.extractfile(im).read())
                elif rel.startswith('arms/') and rel.endswith('.inventory.json'):
                    (out / 'inventory').mkdir(exist_ok=True)
                    (out / 'inventory' / Path(rel).name).write_bytes(data)
                elif rel.startswith('canonical/') and rel.endswith('.json.gz'):
                    (out / 'canonical' / Path(rel).name).write_bytes(data)       # one record per arm
                elif rel.startswith('canonical/') and rel.endswith('.json'):
                    (out / 'canonical' / f'{shard}_{Path(rel).name}').write_bytes(data)  # bundle list form
                elif rel in ('SHARD_STATUS.json', 'ENVIRONMENT.json'):
                    (out / 'status' / f'{shard}_{rel}').write_bytes(data)
        for o in f['status'].get('outcomes', []):
            (done if o.get('returncode') == 0 and o.get('completion_indicator_exists') else failed).append(o['id'])
        report['shards'][shard] = dict(state=f['status'].get('state'), branch=f['ref'], commit=f['commit'],
                                       done=done, failed=failed,
                                       missing=[x for x in expected if x not in done and x not in failed])
    total = sum(len(s['arms']) for s in spec['shards'].values())
    report['arms_done'] = sum(len(v['done']) for v in report['shards'].values())
    report['arms_total'] = total
    (out / 'COLLECTION.json').write_text(json.dumps(report, indent=1) + '\n')
    for shard, v in report['shards'].items():
        print(f"{shard}: {v['state']:>24}  done {len(v['done'])}  failed {len(v.get('failed', []))}  "
              f"missing {len(v['missing'])}  {v.get('branch', '')}")
    print(f"arms done {report['arms_done']}/{total}")


if __name__ == '__main__':
    main()
