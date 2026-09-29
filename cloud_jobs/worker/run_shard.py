#!/usr/bin/env python3
"""Run one shard of a reviewed job inside a Claude Code cloud session.

What is reused unchanged from each reviewed bundle (its launcher/launch.py):
  * validate(root)            - manifest, source and config hashes;
  * validate_protocol(...)    - the bundle's preregistered design gate;
  * require_review(...)       - an approved independent review of this exact manifest;
  * build_command(...)        - the exact engine command and per-arm time budgets;
  * child_environment()       - single-thread numerical libraries, SERVER site.
Only the scheduler is new: a small process pool sized to this container.

Each finished arm is packaged immediately into the session repository:
  <results>/<job>/<shard>/arms/<arm>.tar.gz   small raw files + logs
  <results>/<job>/<shard>/arms/<arm>.inventory.json  sha256/size of every file,
      including large parameter snapshots that are NOT committed
  <results>/<job>/<shard>/canonical/<bundle>.json    canonical v10 per-arm records,
      computed here while the snapshots still exist
Standard library only in this process; no scientific import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tarfile
import time

POLL = 5.0


def utc(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)


def load_launcher(root):
    spec = importlib.util.spec_from_file_location('launch_' + root.name, root / 'launcher' / 'launch.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses in the launcher need a registered module
    spec.loader.exec_module(module)
    return module


def git_head(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def environment_record(jobs_root, python):
    info = dict(utc=utc(), hostname=socket.gethostname(), platform=platform.platform(),
                affinity_cpus=len(os.sched_getaffinity(0)), jobs_commit=git_head(jobs_root),
                runner_sha256=sha256(Path(__file__)), runner_python=sys.version.split()[0],
                engine_python=python)
    try:
        out = subprocess.run([python, '-c', 'import sys,numpy,scipy;print(sys.version.split()[0],'
                              'numpy.__version__,scipy.__version__)'], capture_output=True, text=True,
                             timeout=120).stdout.split()
        info.update(engine_python_version=out[0], numpy=out[1], scipy=out[2])
    except Exception as exc:
        info['engine_versions_error'] = repr(exc)
    try:
        model = [l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                 if l.startswith('model name')]
        info['cpu_model'] = model[0] if model else None
        mem = [l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal')]
        info['memory'] = mem[0] if mem else None
    except OSError:
        pass
    return info


def package_arm(arm, data_dir, logs_dir, out_dir, exclude):
    """Tar small files (+logs); inventory every file with hash, size and packed flag."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    tar_path = out_dir / f'{arm}.tar.gz'
    tmp = tar_path.with_name(tar_path.name + '.tmp')
    with tarfile.open(tmp, 'w:gz', compresslevel=9) as tar:
        for base, prefix in ((data_dir, f'data/{arm}'), (logs_dir, 'logs')):
            if not base.exists():
                continue
            for path in sorted(p for p in base.rglob('*') if p.is_file()):
                rel = path.relative_to(base).as_posix()
                if prefix == 'logs' and not path.name.startswith(arm + '.'):
                    continue
                skip = any(fnmatch.fnmatch(rel, pat) for pat in exclude) and prefix != 'logs'
                inventory.append(dict(path=f'{prefix}/{rel}', bytes=path.stat().st_size,
                                      sha256=sha256(path), packed=not skip))
                if not skip:
                    tar.add(path, arcname=f'{prefix}/{rel}')
    os.replace(tmp, tar_path)
    atomic_json(out_dir / f'{arm}.inventory.json',
                dict(arm=arm, created_utc=utc(), files=inventory, excluded_patterns=exclude,
                     tar_sha256=sha256(tar_path), tar_bytes=tar_path.stat().st_size))


def canonical_records(summarizer, python, manifest_path, exec_dir, arms, out_path, scratch):
    """Run the unchanged canonical summarizer on this bundle; keep this shard's arms."""
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / (out_path.stem + '_SUMMARY.json')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run([python, str(summarizer), '--manifest', str(manifest_path), '--execution',
                           str(exec_dir), '--output', str(target)], capture_output=True, text=True,
                          timeout=1800, env=env)
    record = dict(created_utc=utc(), summarizer=str(summarizer), summarizer_sha256=sha256(summarizer),
                  manifest_sha256=sha256(manifest_path), returncode=proc.returncode,
                  stderr_tail=proc.stderr[-2000:])
    if proc.returncode == 0 and target.exists():
        full = json.loads(target.read_text())
        record['arms'] = [a for a in full['arms'] if a['id'] in arms]
        record['inputs_unchanged'] = full.get('inputs_unchanged')
        record['snapshot_usable'] = full.get('snapshot_usable')
        record['reader_errors_for_shard'] = [e for e in full.get('reader_errors', [])
                                             if any(a in json.dumps(e) for a in arms)]
    atomic_json(out_path, record)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job', required=True)
    ap.add_argument('--shard', required=True)
    ap.add_argument('--jobs-root', required=True, type=Path, help='checkout of the jobs branch')
    ap.add_argument('--results-root', required=True, type=Path, help='cloud_results/ inside the session repo')
    ap.add_argument('--work', type=Path, default=Path('/tmp/cloud_work'))
    ap.add_argument('--python', default=sys.executable, help='pinned interpreter for engines/summarizer')
    ap.add_argument('--max-workers', type=int, default=0)
    ap.add_argument('--global-seconds', type=float, default=0, help='0 = value in SHARDS.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    jobs_root = args.jobs_root.resolve()
    spec = json.loads((jobs_root / 'cloud_jobs' / 'jobs' / args.job / 'SHARDS.json').read_text())
    if args.shard not in spec['shards']:
        raise SystemExit(f'Unknown shard {args.shard}')
    shard = spec['shards'][args.shard]
    leaf = jobs_root / spec['leaf']
    summarizer = jobs_root / spec['summarizer']['path']
    if sha256(summarizer) != spec['summarizer']['sha256']:
        raise SystemExit('Canonical summarizer changed')
    exclude = spec['pack_exclude']
    out = args.results_root.resolve() / args.job / args.shard

    # Validate every bundle this shard touches with its own reviewed launcher code.
    bundles = {}
    for bundle in sorted({b for b, _ in shard['arms']}):
        info = spec['bundles'][bundle]
        root = (leaf / info['path']).resolve()
        launch = load_launcher(root)
        manifest, mhash = launch.validate(root)
        if mhash != info['manifest_sha256']:
            raise SystemExit(f'{bundle}: manifest hash {mhash} differs from SHARDS.json')
        launch.validate_protocol(manifest, root)
        launch.require_review(root, mhash)
        bundles[bundle] = dict(root=root, launch=launch, manifest=manifest, hash=mhash,
                               entries={e['id']: e for e in manifest['configs']},
                               exec_dir=args.work / args.job / args.shard / bundle / info['execution_dir'])
    arms = []
    for bundle, arm in shard['arms']:
        if arm not in bundles[bundle]['entries']:
            raise SystemExit(f'{arm} not in bundle {bundle}')
        arms.append((bundle, arm))
    cpus = len(os.sched_getaffinity(0))
    workers = max(1, min(args.max_workers or cpus, cpus, len(arms)))
    global_seconds = args.global_seconds or float(spec['global_seconds'])
    plan = dict(job=args.job, shard=args.shard, arms=[a for _, a in arms], workers=workers, cpus=cpus,
                per_arm_seconds={b: v['manifest']['runtime']['per_arm_seconds'] for b, v in bundles.items()},
                global_seconds=global_seconds, expected_minutes=shard.get('expected_minutes'),
                manifests={b: v['hash'] for b, v in bundles.items()}, results=str(out))
    print(json.dumps(plan, indent=1))
    if args.dry_run:
        return 0
    if (out / 'SHARD_STATUS.json').exists():
        raise SystemExit('Results for this shard already exist in this repository; never rerun a shard in place')
    for v in bundles.values():
        if v['exec_dir'].exists():
            raise SystemExit(f"Refusing to reuse execution directory {v['exec_dir']}")
        (v['exec_dir'] / 'data').mkdir(parents=True)
        (v['exec_dir'] / 'logs').mkdir()

    os.environ['AGOP_EXECUTION_SITE'] = 'SERVER'  # cloud container, never the personal machine
    if os.nice(0) < 10:
        os.nice(10 - os.nice(0))
    env_record = environment_record(jobs_root, args.python)
    env_record['plan'] = plan
    atomic_json(out / 'ENVIRONMENT.json', env_record)

    pending = list(arms)
    active, done = {}, []
    start = time.monotonic()
    global_deadline = start + global_seconds
    stop = [None]

    def handler(signum, frame):
        stop[0] = signum

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    def exec_status(bundle):
        v = bundles[bundle]
        mine = [a for b, a in arms if b == bundle]
        atomic_json(v['exec_dir'] / 'STATUS.json', dict(
            status='running' if (pending or active) else 'completed', updated_utc=utc(),
            pending=[a for b, a in pending if b == bundle],
            active=[dict(id=a) for (b, a) in active if b == bundle],
            completed=[d for d in done if d['bundle'] == bundle], shard_arms=mine,
            scientific_success='not_assessed'))

    def shard_status(state):
        atomic_json(out / 'SHARD_STATUS.json', dict(
            job=args.job, shard=args.shard, state=state, updated_utc=utc(),
            elapsed_seconds=time.monotonic() - start, workers=workers,
            arms_total=len(arms), arms_done=len(done), arms_active=[a for _, a in active],
            arms_pending=[a for _, a in pending], outcomes=done))
        for b in bundles:
            exec_status(b)

    def finish(key, child, code):
        bundle, arm = key
        v = bundles[bundle]
        child['out'].close()
        child['err'].close()
        indicator = v['launch'].ENGINES[v['entries'][arm]['engine']][1]
        data = v['exec_dir'] / 'data' / arm
        record = dict(id=arm, bundle=bundle, engine=v['entries'][arm]['engine'], returncode=code,
                      elapsed_seconds=time.monotonic() - child['started'], stop_reason=child['reason'],
                      term_sent=child['term'], kill_sent=child['kill'], finished_utc=utc(),
                      completion_indicator=indicator, completion_indicator_exists=(data / indicator).is_file())
        done.append(record)
        del active[key]
        package_arm(arm, data, v['exec_dir'] / 'logs', out / 'arms', exclude)
        exec_status(bundle)
        finished_here = [d['id'] for d in done if d['bundle'] == bundle]
        canonical_records(summarizer, args.python, v['root'] / 'MANIFEST.json', v['exec_dir'],
                          finished_here, out / 'canonical' / f'{bundle}.json', args.work / 'summaries')
        shard_status('running')

    shard_status('running')
    try:
        while pending or active:
            now = time.monotonic()
            for key, child in list(active.items()):
                code = child['proc'].poll()
                if code is not None:
                    try:
                        os.killpg(child['proc'].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    finish(key, child, code)
                    continue
                deadline = min(child['deadline'], global_deadline)
                if stop[0] and not child['term']:
                    child['reason'] = child['reason'] or 'runner_signal'
                if (now >= deadline - 8 or stop[0]) and not child['term']:
                    child['reason'] = child['reason'] or ('global_wall_cap' if global_deadline <= child['deadline']
                                                          else 'per_arm_wall_cap')
                    os.killpg(child['proc'].pid, signal.SIGTERM)
                    child['term'], child['term_at'] = True, now
                if not child['kill'] and (now >= deadline - 2 or
                                          (stop[0] and child['term'] and now >= child['term_at'] + 10)):
                    os.killpg(child['proc'].pid, signal.SIGKILL)
                    child['kill'] = True
            while pending and len(active) < workers and not stop[0]:
                bundle, arm = pending[0]
                v = bundles[bundle]
                runtime = v['manifest']['runtime']
                now = time.monotonic()
                available = min(float(runtime['per_arm_seconds']), global_deadline - now)
                if available <= float(runtime['diagnostic_reserve_seconds']) + 10:
                    break
                entry = v['entries'][arm]
                cutoff = utc(time.time() + available - 10)
                command = v['launch'].build_command(v['root'], v['exec_dir'], entry, runtime, available, cutoff)
                command[0] = args.python
                logs = v['exec_dir'] / 'logs'
                out_f = open(logs / f'{arm}.stdout.log', 'xb')
                err_f = open(logs / f'{arm}.stderr.log', 'xb')
                proc = subprocess.Popen(command, cwd=v['root'], env=v['launch'].child_environment(),
                                        stdout=out_f, stderr=err_f, start_new_session=True)
                active[(bundle, arm)] = dict(proc=proc, out=out_f, err=err_f, started=now,
                                             deadline=now + float(runtime['per_arm_seconds']), reason=None,
                                             term=False, kill=False, term_at=0.0, command=command)
                pending.pop(0)
                shard_status('running')
            if not active and pending and (stop[0] or global_deadline - time.monotonic() < 200):
                break
            time.sleep(POLL)
    finally:
        for key, child in list(active.items()):
            try:
                os.killpg(child['proc'].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child['proc'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            child['reason'] = child['reason'] or 'runner_cleanup'
            finish(key, child, child['proc'].poll())
        state = 'finished' if not pending else ('interrupted' if stop[0] else 'global_budget_exhausted')
        shard_status(state)
    print(json.dumps(dict(state=state, done=len(done), pending=[a for _, a in pending])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
