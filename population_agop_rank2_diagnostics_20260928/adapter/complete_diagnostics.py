#!/usr/bin/env python3
"""Complete diagnostics on pinned original snapshots. Never train or rewrite inputs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from planner import verify_inputs, make_plan, complete_rows, sha256

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT/'swiglu/code'
sys.path.insert(0,str(ROOT/'launcher'))
import launch


def clean(value):
    # Only called after SERVER/provenance gates and explicit lazy numpy import.
    import numpy as np
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(float,np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value


def atomic_json(path,value):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(clean(value),stream,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-seconds',type=float,required=True)
    parser.add_argument('--deadline-utc',required=True)
    args=parser.parse_args(argv)
    # Direct invocation must not bypass scheduler admission or review gates.
    launch.require_server()
    manifest,digest=launch.validate(ROOT)
    launch.validate_protocol(manifest,ROOT)
    launch.require_review(ROOT,digest)
    launch.require_capacity(manifest['runtime'])
    launch.require_priority()
    for key in launch.THREAD_VARIABLES:
        if os.environ.get(key) != '1': raise RuntimeError('Single-thread environment required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU-only environment required')
    if os.environ.get('AGOP_DIAGNOSTIC_LAUNCHER_PID') != str(os.getppid()):
        raise RuntimeError('Use the reviewed scheduler; direct scientific execution refused')
    entry=manifest['configs'][0]
    if args.tag != entry['id'] or args.config.resolve() != (ROOT/entry['config_path']).resolve():
        raise ValueError('Only the pinned case/config is supported')
    if not math.isfinite(args.diagnostic_seconds) or not 0 < args.diagnostic_seconds <= 1200:
        raise ValueError('Diagnostic budget must be positive and at most 1200 seconds')
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    if deadline.tzinfo is None: raise ValueError('Deadline requires timezone')
    output=args.out_dir.resolve()
    if output.parent.name != 'data' or output.parent.parent.parent != ROOT:
        raise ValueError('Output must be inside a fresh scheduler execution/data directory')
    if output.name != entry['id']: raise ValueError('Output case mismatch')
    if output.exists(): raise FileExistsError('Preserve existing output; use a new attempt')
    spec=launch.load_json(ROOT/'INPUT_SPEC.json')
    source=Path(spec['remote_input_root'])
    inputs_before=verify_inputs(spec,source)
    raw=launch.load_json(source/spec['raw_name'])
    original=launch.load_json(source/'result.json')
    partial=launch.load_json(source/'diagnostics_partial.json')
    computed,plan=make_plan(raw,original,partial,spec)
    frozen=launch.load_json(CODE/'SOURCE_MANIFEST.json')
    if frozen.get('status') != 'frozen_after_review' or sha256(CODE/'SOURCE_MANIFEST.json') != spec['source_manifest_sha256']:
        raise ValueError('Frozen source gate mismatch')
    for name,expected in spec['source_files_sha256'].items():
        if sha256(CODE/name) != expected: raise ValueError('Scientific source changed: '+name)
    if plan != launch.load_json(ROOT/'COMPLETION_PLAN.json'):
        raise ValueError('Runtime completion plan differs from reviewed original grid')
    # No scientific imports before all server, source, config, review and input gates.
    import numpy as np
    sys.path.insert(0,str(CODE/'engine'));sys.path.insert(0,str(CODE))
    import diagnostics
    output.mkdir(exist_ok=False)
    receipt=dict(artifact_kind='derived_diagnostics_sidecar',tag=args.tag,
        original_process_diagnostics_complete=False,original_result_completed=original['completed'],
        original_success_count=len(plan['preserved_success_indices']),original_checkpoint_count=len(raw['rows']),
        original_training_termination=original['termination'],original_input_root=str(source),
        cfg=original['cfg'],cell=original['cell'],rank=original['rank'],
        original_all_update_count=original['all_update_count'],
        input_files_before=inputs_before,source_manifest_sha256=spec['source_manifest_sha256'],
        manifest_sha256=digest,source_sha256=manifest['source_sha256'],
        config_sha256=sha256(args.config),plan=plan,new_training_updates=0,
        diagnostic_schema_version=diagnostics.DIAGNOSTIC_SCHEMA_VERSION,
        agop_resolution_guard=diagnostics.AGOP_RESOLUTION_GUARD,
        quadrature_multiplier=1,independent_seed=False,
        started_utc=launch.utc(),diagnostic_budget_seconds=args.diagnostic_seconds,
        global_deadline_utc=args.deadline_utc,effective_nice=os.nice(0),
        thread_environment={k:os.environ[k] for k in launch.THREAD_VARIABLES},
        input_files_unchanged_after=None,completion_pass_finished=False,
        completion_pass_complete=False,original_result_mutated=False)
    start=time.monotonic()
    end=start+min(args.diagnostic_seconds,max(0,deadline.timestamp()-time.time()-10))
    stop={'signal':None}
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in previous: signal.signal(sig,lambda signum,frame:stop.update(signal=signum))
    def persist(rows,origins,attempted):
        receipt.update(diagnostics=rows,diagnostic_row_origins=origins,
            attempted_indices=list(attempted),diagnostic_success_count=sum(r['status']=='ok' for r in rows),
            missing_indices=[i for i,r in enumerate(rows) if r['status']!='ok'],
            updated_utc=launch.utc(),diagnostic_wall_seconds=time.monotonic()-start,
            interruption_signal=stop['signal'])
        atomic_json(output/'DIAGNOSTICS_COMPLETION.json',receipt)
    persist(computed,plan['origins'],[])
    exit_code=1
    try:
        with np.load(source/spec['snapshots_name'],allow_pickle=False) as data:
            arrays={name:data[name] for name in ('t','P','V','a')}
        count=len(raw['rows']);cfg=spec['cfg']
        shapes={'t':(count,),'P':(count,cfg['d']+1,cfg['m']),
                'V':(count,cfg['d']+1,cfg['m']),'a':(count,cfg['m'])}
        for name,arr in arrays.items():
            if arr.shape != shapes[name] or not np.all(np.isfinite(arr)):
                raise ValueError('Saved array shape/finiteness mismatch: '+name)
            arr.flags.writeable=False
        if not np.array_equal(arrays['t'],np.array([r['t'] for r in raw['rows']])):
            raise ValueError('Snapshot timestamps differ from original checkpoint grid')
        if time.monotonic() >= end: raise TimeoutError('Diagnostic budget exhausted before engine setup')
        engine=diagnostics.make_engine(cfg)
        computed,origins,attempted=complete_rows(computed,plan,
            lambda i:diagnostics.metrics(engine,arrays['P'][i],arrays['V'][i],arrays['a'][i]),
            persist,lambda:stop['signal'] is not None or time.monotonic()>=end)
        receipt['input_files_unchanged_after']=False
        inputs_after=verify_inputs(spec,source)
        launch.validate(ROOT,digest)
        receipt.update(input_files_after=inputs_after,input_files_unchanged_after=True,
            completion_pass_finished=True,completion_pass_complete=all(r['status']=='ok' for r in computed),
            ended_utc=launch.utc(),stop_reason=('signal' if stop['signal'] else
               'complete_grid' if all(r['status']=='ok' for r in computed) else 'budget_or_failed_states'))
        persist(computed,origins,attempted)
        exit_code=0 if receipt['completion_pass_complete'] else 124
    except Exception as exc:
        receipt.update(completion_pass_finished=True,completion_pass_complete=False,
            exception=f'{type(exc).__name__}: {exc}',ended_utc=launch.utc())
        atomic_json(output/'DIAGNOSTICS_COMPLETION.json',receipt)
        raise
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)
    return exit_code

if __name__=='__main__':
    raise SystemExit(main())
