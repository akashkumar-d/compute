"""Small deterministic fixed-checkpoint audit; no training or old-file writes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
from stable_relu import population_agop,fixed_checkpoint_svd,refit_uncertainty_report,spectral_summary

WROOT=Path(__file__).resolve().parents[2]
B3=WROOT/'broader_teachers_dataprep_v3'
OUT=Path(__file__).resolve().parents[1]/'review'/'SAVED_DIAGNOSTIC_AUDIT.json'
sys.path.insert(0,str(B3/'code'))
from population import moments

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def locate(name):
    hits=list(B3.glob('remote*final/experiments/*/**/data/'+name+'/meta.json'))
    if len(hits)!=1:raise ValueError((name,hits))
    return hits[0].parent
def loadrows(p):return [json.loads(s) for s in (p/'rows.jsonl').read_text().splitlines()]
def save(report):
    OUT.parent.mkdir(exist_ok=True,parents=True)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def main():
    cases=[]
    for link,choice in [('abs','last'),('h3','first_masked_plateau'),('h2','initial'),('h2','plateau_end')]:
        name='relu_'+link+'_r8_d64_m128_s1e-4_seed501';p=locate(name);rows=loadrows(p)
        if choice=='last':row=rows[-1]
        elif choice=='initial':row=rows[0]
        elif choice=='plateau_end':row=max((x for x in rows if x['in_plateau']),key=lambda x:x['step'])
        else:
            candidates=[x for x in rows if x['step']>0 and x['in_plateau'] and not x['agop_valid']]
            row=candidates[0] if candidates else max((x for x in rows if x['in_plateau']),key=lambda x:x['step'])
        cases.append((name,p,row,choice))
    report=dict(scope='Selected saved checkpoints only; no training, no event reclassification',
        tests_passed=7,states=[],source_sha256=sha(Path(__file__).with_name('stable_relu.py')))
    save(report)
    for name,p,row,choice in cases:
        state=p/'states'/('step%08d.npz'%row['step']);before=sha(state)
        z=np.load(state);A,W,b=[z[k] for k in ('A','W','b')];U=np.eye(W.shape[1])[:,:8]
        start=time.perf_counter();G,new,top=population_agop(A,W,b,U)
        Q=moments(W,b,0.)[1];old=W.T@(Q*np.outer(A,A))@W/len(A)**2
        old_info,_=spectral_summary(old,U)
        # Saved byte-exact runner scaling can alter cancellation artifacts.
        old_info['saved_raw_alignment']=[row['A_min'],row['A_mean']]
        old_info['saved_legacy_valid']=row['agop_valid']
        audit=fixed_checkpoint_svd(A,W,b,U,sample_sizes=(2048,),seeds=(8101,8102),population_G=G)
        result=dict(run=name,selection=choice,step=row['step'],in_original_plateau=row['in_plateau'],
            checkpoint_path=str(state),checkpoint_sha256=before,
            old=old_info,new=new,independent=audit,
            old_new_difference_operator=float(np.linalg.norm(old-G,2)),
            refit=refit_uncertainty_report(row['refit_info']),elapsed_seconds=time.perf_counter()-start)
        assert before==sha(state)
        report['states'].append(result);save(report)
        print(json.dumps(dict(run=name,step=row['step'],elapsed=result['elapsed_seconds'],
            old_saved=[row['A_min'],row['A_mean']],new=[new['A_min'],new['A_mean']],
            new_rank=new['numerical_rank'],new_screen=new['numerical_screen_passed'],
            warning_count=new['quadrature_warning_count'])),flush=True)
    report['complete']=True;save(report)

if __name__=='__main__':main()
