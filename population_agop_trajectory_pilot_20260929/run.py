"""Explicit one-worker entry point. No scientific imports until all gates pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
NAME=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def validate(root=HERE):
    manifest=read(root/'MANIFEST.json');mh=sha(root/'MANIFEST.json')
    for name,expected in manifest['pinned_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=expected:
            raise ValueError('Changed/unsafe pinned source: '+name)
    plan=read(root/'MATCHED_SCALE_PROTOCOL.json');settings=read(root/'SETTINGS.json')
    for arm in plan['arms']:
        if read(root/'configs'/(arm['id']+'.json'))!=arm:
            raise ValueError('Configuration differs from matched protocol')
    if settings['matched_protocol_sha256']!=sha(root/'MATCHED_SCALE_PROTOCOL.json'):
        raise ValueError('Matched protocol hash mismatch')
    for name,expected in read(root/'FROZEN_SOURCES.json')['source_sha256'].items():
        if sha(root/name)!=expected: raise ValueError('Frozen source changed: '+name)
    return manifest,mh,plan,settings


def require_review(root,mh):
    r=read(root/'REVIEW.json')
    if r.get('status')!='approved_for_execution' or r.get('manifest_sha256')!=mh or not r.get('reviewer'):
        raise RuntimeError('Independent source approval of the exact manifest is required')


def require_smoke(root,mh,output):
    if not NAME.fullmatch(output): raise ValueError('Invalid smoke directory name')
    result=root/output/'RESULTS.json';r=read(result);review=read(root/'SMOKE_REVIEW.json')
    if (r.get('manifest_sha256')!=mh or r.get('mode')!='smoke' or r.get('status')!='complete'
        or r.get('all_checks_passed') is not True or r.get('inputs_unchanged') is not True
        or len(r.get('arms',[]))!=4 or len(r.get('anchors',[]))!=4
        or any(x.get('status')!='passed' for x in r['arms']+r['anchors'])):
        raise RuntimeError('Complete successful four-arm/four-anchor smoke required')
    if (review.get('status')!='approved_for_pilot' or review.get('manifest_sha256')!=mh
        or review.get('smoke_result_sha256')!=sha(result) or not review.get('reviewer')):
        raise RuntimeError('Independent exact smoke review required before pilot training')


def runtime_inputs(root):
    plan=read(root/'RUNTIME_INPUTS.json')
    for name,expected in plan['files'].items():
        if Path(name).name!=name or sha(root/'runtime_inputs'/name)!=expected:
            raise ValueError('Missing/changed smoke reference array: '+name)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['smoke','arm']);ap.add_argument('--arm')
    ap.add_argument('--output',required=True);ap.add_argument('--cpu',type=int,required=True)
    ap.add_argument('--other-workers',type=int,required=True,
        help='Parent-verified live task workers excluding this one, not an automatic discovery claim')
    ap.add_argument('--smoke-output');ap.add_argument('--dry-run',action='store_true');args=ap.parse_args(argv)
    manifest,mh,plan,settings=validate()
    if not NAME.fullmatch(args.output): raise ValueError('Fresh direct-child output name required')
    out=HERE/args.output
    if out.exists(): raise FileExistsError('Output already exists; inspect preserved prior attempt')
    if not 0<=args.other_workers<=29: raise ValueError('One worker plus other live workers must be <=30')
    arms={x['id']:x for x in plan['arms']}
    if args.mode=='arm' and args.arm not in arms: raise ValueError('Exactly one declared arm required')
    if args.mode=='smoke' and args.arm is not None: raise ValueError('Smoke always retains all four arms')
    if args.dry_run:
        print(json.dumps(dict(valid=True,manifest_sha256=mh,mode=args.mode,model_evaluations=0,
            output_created=False,scientific_imports=False)));return 0
    import probe_resources as guard
    guard.require_server();require_review(HERE,mh)
    if args.mode=='smoke': runtime_inputs(HERE)
    else:
        if not args.smoke_output: raise ValueError('--smoke-output required for pilot')
        require_smoke(HERE,mh,args.smoke_output)
    resources=guard.require_capacity(dict(workers=1,reserved_cpus=31,
        min_available_memory_gib=8,min_free_disk_gib=5))
    guard.require_priority()
    if args.cpu not in os.sched_getaffinity(0): raise ValueError('Requested CPU outside allocated affinity')
    os.sched_setaffinity(0,{args.cpu})
    if os.sched_getaffinity(0)!={args.cpu}: raise RuntimeError('Single-CPU affinity failed')
    for k in guard.THREAD_VARIABLES: os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    # Conservative persistent journal. No automatic restart, stale-lock deletion,
    # overwrite, or reuse of an attempted arm after a timeout/interruption.
    key='smoke' if args.mode=='smoke' else args.arm
    journal=HERE/('.attempt_'+key+'.json')
    receipt=dict(mode=args.mode,arm=args.arm,output=args.output,pid=os.getpid(),manifest_sha256=mh,
        cpu=args.cpu,other_workers_parent_verified=args.other_workers,
        policy='Never delete/reuse automatically; new attempts need independent recovery review')
    with journal.open('x') as f: json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    out.mkdir();started=time.monotonic();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    report=dict(status='started',mode=args.mode,manifest_sha256=mh,resources=resources,
        dispatch=receipt,model_evaluations=0,accepted_updates=0,method_constant=True,
        no_fresh_seed_claim=True,scope='Sampled numerical safeguards, not rigorous trajectory certification')
    # Exactly these imports start scientific availability, after all guards.
    sys.path.insert(0,str(HERE/'engine'))
    from runtime import Session,run_smoke,run_arm,BudgetStop
    session=Session(HERE,out,settings,started,stopped,report);session.save()
    try:
        if args.mode=='smoke': run_smoke(session,plan)
        else: run_arm(session,arms[args.arm],plan['criteria'])
    except BudgetStop as exc: report.update(status='censored',error=str(exc))
    except Exception as exc: report.update(status='failed',error=repr(exc))
    finally:
        try: report['inputs_unchanged']=all(sha(HERE/k)==v for k,v in manifest['pinned_sha256'].items())
        except OSError as exc: report.update(inputs_unchanged=False,input_error=repr(exc))
        if args.mode=='smoke':
            try: runtime_inputs(HERE)
            except Exception as exc: report.update(inputs_unchanged=False,input_error=repr(exc))
        report['termination_requested']=stopped[0];session.save()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__=='__main__': raise SystemExit(main())
