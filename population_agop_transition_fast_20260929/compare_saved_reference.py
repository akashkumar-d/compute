"""Compare downloaded fixed-state outputs only; never import or evaluate models."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    ref={p.name:p for p in a.reference.glob('*.npz')}
    cand={p.name:p for p in a.candidate.glob('*.npz')}
    report=dict(scope='Common saved fixed-state arrays only; reference censoring remains unresolved',
        rtol=1e-12,atol=1e-13,reference_files=sorted(ref),candidate_files=sorted(cand),
        missing_candidate_files=sorted(set(ref)-set(cand)),
        candidate_files_without_reference=sorted(set(cand)-set(ref)),arrays=[],loss_calls=[])
    for name in sorted(set(ref)&set(cand)):
        with np.load(ref[name],allow_pickle=False) as x,np.load(cand[name],allow_pickle=False) as y:
            if set(x.files)!=set(y.files):
                raise ValueError('Saved array keys differ: '+name)
            for key in x.files:
                if x[key].shape!=y[key].shape:
                    raise ValueError('Saved array shapes differ: '+name+'/'+key)
                exact=x[key].dtype==y[key].dtype and x[key].tobytes()==y[key].tobytes()
                diff=float(np.max(np.abs(x[key]-y[key]))) if x[key].size else 0.
                close=bool(np.allclose(x[key],y[key],rtol=1e-12,atol=1e-13,equal_nan=False))
                report['arrays'].append(dict(file=name,key=key,exact_bytes=exact,max_abs_difference=diff,
                    allclose=close,reference_sha256=sha(ref[name]),candidate_sha256=sha(cand[name])))
    def calls(path):
        rows=[json.loads(x) for x in (path/'EVALUATIONS.jsonl').read_text().splitlines()]
        done=[r for r in rows if r['status']=='returned']
        if len({r['label'] for r in done})!=len(done):
            raise ValueError('Duplicate returned call label')
        return {r['label']:r for r in done}
    old,new=calls(a.reference),calls(a.candidate)
    report['missing_candidate_returned_calls']=sorted(set(old)-set(new))
    report['candidate_calls_without_reference']=sorted(set(new)-set(old))
    for label in sorted(set(old)&set(new)):
        x,y=old[label],new[label]
        report['loss_calls'].append(dict(label=label,loss_difference=y['L']-x['L'],
            exactly_equal=y['L']==x['L'],
            allclose=bool(np.isclose(y['L'],x['L'],rtol=1e-12,atol=1e-13)),
            reference_seconds=x['wall_seconds'],candidate_seconds=y['wall_seconds'],
            measured_speed_ratio=x['wall_seconds']/y['wall_seconds']))
    rows=report['arrays']+report['loss_calls']
    ok=bool(rows) and not report['missing_candidate_files'] and not report['missing_candidate_returned_calls'] and all(x['allclose'] for x in rows)
    report['status']='PASS_common_recorded_values' if ok else 'FAIL_or_missing_candidate'
    report['reference_completion']=json.loads((a.reference/'RESULTS.json').read_text())['status']
    report['candidate_completion']=json.loads((a.candidate/'RESULTS.json').read_text())['status']
    report['all_common_values_bitwise_equal']=all(x.get('exact_bytes',x.get('exactly_equal')) for x in rows)
    report['model_evaluations']=0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','all_common_values_bitwise_equal','reference_completion','candidate_completion')}))
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())

