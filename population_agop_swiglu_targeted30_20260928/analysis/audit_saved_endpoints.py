"""Bounded, read-only order-doubling audit of declared completed endpoints."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,record):
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    temporary.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tags',nargs='+',required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Preserve previous audit')
    code=Path(__file__).resolve().parents[1]/'bundle/swiglu/code';sys.path.insert(0,str(code));import diagnostics
    args.out.parent.mkdir(parents=True,exist_ok=True)
    record=dict(scope='Declared initialization and terminal states of completed runs, not trajectory certification',no_training=True,selected_tags=args.tags,inputs={},rows=[],complete=False)
    started=time.monotonic()
    for tag in args.tags:
        folder=args.execution/'data'/tag;receipt=folder/'result.json';raw=folder/(tag+'.json');snap=folder/(tag+'_snaps.npz')
        for p in [receipt,raw,snap]:record['inputs'][str(p)]=sha(p)
        assert json.loads(receipt.read_text())['completed'] is True
        data=json.loads(raw.read_text());cfg=data['cfg'];states=np.load(snap)
        assert len(states['t'])==len(data['rows'])
        engines={factor:diagnostics.make_engine(cfg,factor) for factor in [1,2]}
        for i in sorted({0,len(states['t'])-1}):
            P,V,a=states['P'][i],states['V'][i],states['a'][i]
            assert np.isclose(states['t'][i],data['rows'][i]['t'],rtol=0,atol=1e-9)
            row=dict(tag=tag,index=i,step=data['rows'][i]['step'],time=float(states['t'][i]),orders={})
            gradients={}; matrices={}
            for factor,E in engines.items():
                ev=E.evaluate(P,V,a,need_grad=True)
                gradients[factor]=np.concatenate([np.ravel(ev[key]) for key in ['gP','gV','ga']])
                matrices[factor]=E.agop(P,V,a,ev)
                row['orders'][str(factor)]=dict(raw_loss=float(ev['L']),gradient_norm=float(np.linalg.norm(gradients[factor])),metrics=diagnostics.metrics(E,P,V,a))
            row['gradient_absolute_difference']=float(np.linalg.norm(gradients[2]-gradients[1]))
            row['gradient_relative_difference']=row['gradient_absolute_difference']/max(float(np.linalg.norm(gradients[2])),1e-300)
            row['agop_relative_frobenius_difference']=float(np.linalg.norm(matrices[2]-matrices[1]))/max(float(np.linalg.norm(matrices[2])),1e-300)
            row['loss_absolute_difference']=abs(row['orders']['2']['raw_loss']-row['orders']['1']['raw_loss'])
            row['Amin_absolute_difference']=abs(row['orders']['2']['metrics']['agop_Amin']-row['orders']['1']['metrics']['agop_Amin'])
            record['rows'].append(row)
            save(args.out,record)
    record['inputs_unchanged']=all(sha(Path(p))==h for p,h in record['inputs'].items())
    record.update(complete=True,wall_seconds=time.monotonic()-started,script_sha256=sha(Path(__file__)))
    save(args.out,record)
    print(json.dumps(dict(rows=len(record['rows']),inputs_unchanged=record['inputs_unchanged'],wall_seconds=record['wall_seconds'],no_training=True)))
if __name__=='__main__':main()
