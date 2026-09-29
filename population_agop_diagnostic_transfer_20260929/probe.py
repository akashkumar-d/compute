"""Compare existing AGOP/refit diagnostics at four saved states; no training."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=HERE/args.output
    if out.parent!=HERE or out.exists():raise ValueError('Fresh direct-child output required')
    pp=HERE/'PROBE_MANIFEST.json';plan=read(pp)
    for n,h in plan['input_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('Changed pinned input: '+n)
    review=read(HERE/'PROBE_REVIEW.json')
    if review.get('status')!='approved_for_execution' or review.get('manifest_sha256')!=sha(pp):
        raise ValueError('Exact source review required')
    import probe_resources as guard
    guard.require_server();resources=guard.require_capacity(dict(workers=1,reserved_cpus=31,
        min_available_memory_gib=8,min_free_disk_gib=5));guard.require_priority()
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    for k in guard.THREAD_VARIABLES:os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    out.mkdir();start=time.monotonic();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    report=dict(status='started',resources=resources,manifest_sha256=sha(pp),model_evaluations=0,
                new_training_updates=0,states=[],scope='Four fixed stored states; no trajectory or learning-success claim')

    def write():
        report['elapsed_seconds']=time.monotonic()-start
        p=out/'RESULTS.json.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(out/'RESULTS.json')

    def evaluate(E,theta,label):
        if stopped[0] or time.monotonic()-start>=350 or report['model_evaluations']>=8:
            raise RuntimeError('Diagnostic cap reached')
        report['model_evaluations']+=1;t0=time.monotonic()
        with (out/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'],label=label,status='started'))+'\n')
        ev=E.evaluate(*theta,need_grad=True);elapsed=time.monotonic()-t0
        with (out/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'],label=label,status='returned',
                L=float(ev['L']),wall_seconds=elapsed))+'\n')
        return ev,elapsed

    write()
    try:
        import numpy as np
        sys.path.insert(0,str(HERE/'engine'))
        import swsmall_periodic as engine
        import diagnostics
        from student_transition_fast import BatchedTransitionSwiGLUPop as Reference
        from hermite_pairs import make_hermite_evaluator_class
        Candidate=make_hermite_evaluator_class(Reference)
        cfg=read(HERE/'probe_inputs/config.json')[0]['args'];r=len(cfg['c'])
        link,breaks=engine.make_link(cfg['link'])
        T=engine.swpop.Teacher(link,np.asarray(cfg['c']),breaks=breaks,n_x=cfg['n_x'])

        class Cached:
            def __init__(self,E,theta,ev):self.E,self.theta,self.ev=E,theta,ev
            def __getattr__(self,name):return getattr(self.E,name)
            def evaluate(self,P,V,a,need_grad=True):
                if not need_grad or not all(np.array_equal(x,y) for x,y in zip((P,V,a),self.theta)):
                    raise ValueError('Cached metric request changed state')
                return self.ev

        def risk(ev,w):
            G=(ev['G']+ev['G'].T)/2;t=ev['t']
            return float((ev['V']-2*t@w+w@G@w)/ev['V'])

        def weights(ev):
            G=(ev['G']+ev['G'].T)/2;t=ev['t'];scale=np.sqrt(np.maximum(np.diag(G),1e-300))
            vals,Q=np.linalg.eigh(G/scale[:,None]/scale[None,:]);q=Q.T@(t/scale);out={};masks={}
            for cut in diagnostics.CUTOFFS:
                keep=vals>cut*vals[-1];z=np.zeros_like(q);z[keep]=q[keep]/vals[keep]
                out[str(cut)]=Q@z/scale;masks[str(cut)]=keep
            return out,vals,masks

        def point_screen(m):
            risks=[z['actual_mse'] for z in m['pinv'].values()]
            residuals=[z['relative_normal_residual'] for z in m['pinv'].values()]
            return dict(agop_resolved=bool(m['agop_full_rank_resolved']),
                refit_stable=bool(all(np.isfinite(risks)) and all(np.isfinite(residuals)) and m['equilibrated_lambda_ratio']>0 and min(risks)>=-1e-8
                                  and max(risks)-min(risks)<=1e-7 and max(residuals)<=1e-7),
                cutoff_spread=float(max(risks)-min(risks)),max_normal_residual=float(max(residuals)))

        def norm(parts):return float(np.sqrt(sum(np.sum(x*x) for x in parts)))

        with np.load(HERE/'probe_inputs/states.npz',allow_pickle=False) as f:
            states=[(step,tuple(f[k][i].copy() for k in ('P','V','a'))) for i,step in enumerate((472,473,2260,2261))]
        for step,theta in states:
            item=dict(step=step,status='running',evaluations={});report['states'].append(item);write()
            records={}
            for name,klass,extra in [('reference',Reference,{}),('candidate',Candidate,dict(hermite_degree=128,hermite_quad_order=24))]:
                E=klass(cfg['d'],T,alpha=cfg['alpha'],intercept='refit',bias=True,quad_order=16,quad_limit=12.,**extra)
                ev,elapsed=evaluate(E,theta,f'{step}/{name}')
                metrics=diagnostics.metrics(Cached(E,theta,ev),*theta)
                M=E.agop(*theta,ev=ev);ws,gspec,masks=weights(ev)
                arrays={k:ev[k] for k in ('L','G','t','gP','gV','ga')}
                arrays['teacher_variance']=ev['V']
                arrays.update({k:z for k,z in zip(('P','V','a'),theta)})
                arrays.update({'Cp_'+k:v for k,v in ev['Cp'].items()})
                arrays.update({'D_'+k:v for k,v in ev['D'].items()})
                arrays.update({'refit_w_'+k:v for k,v in ws.items()})
                arrays.update({'refit_mask_'+k:v for k,v in masks.items()})
                arrays['equilibrated_gram_eigenvalues']=gspec;arrays['AGOP']=M
                if name=='candidate':
                    hd=ev['hermite_diagnostics'];bank=hd['scalar_bank']
                    arrays.update({'tail_'+k:v for k,v in hd['per_pair_tail_estimates'].items()})
                    arrays.update({'bank_'+k:v for k,v in bank.items() if isinstance(v,np.ndarray)})
                np.savez_compressed(out/f'step{step}_{name}.npz',**arrays)
                item['evaluations'][name]=dict(loss=float(ev['L']),wall_seconds=elapsed,metrics=metrics,
                    point_screen=point_screen(metrics),scalar_energy_valid=True if name=='reference' else bool(hd['all_scalar_estimates_valid']))
                records[name]=(ev,M,metrics,ws);write()
            er,Mr,mr,wr=records['reference'];eh,Mh,mh,wh=records['candidate']
            dM=Mh-Mr;frel=float(np.linalg.norm(dM)/max(np.linalg.norm(Mr),1e-300))
            err2=float(np.linalg.norm(dM,2));evals=np.linalg.eigvalsh((Mr+Mr.T)/2)[::-1]
            gap=float(evals[r-1]-evals[r]);gap_condition=bool(mr['agop_full_rank_resolved'] and err2<=1e-3*gap)
            Qr=np.linalg.eigh((Mr+Mr.T)/2)[1][:,-r:];Qh=np.linalg.eigh((Mh+Mh.T)/2)[1][:,-r:]
            projector_difference=float(np.linalg.norm(Qh@Qh.T-Qr@Qr.T,2))
            refit={}
            for cut in diagnostics.CUTOFFS:
                k=str(cut);riskr=risk(er,wr[k]);riskh=risk(eh,wh[k]);cross=risk(er,wh[k]);reverse=risk(eh,wr[k])
                refit[k]=dict(reference_self=riskr,candidate_self=riskh,candidate_on_reference=cross,
                    reference_on_candidate=reverse,self_difference=abs(riskh-riskr),
                    same_head_risk_error=abs(cross-riskh),reverse_same_head_risk_error=abs(reverse-riskr),
                    candidate_excess_on_reference=cross-riskr,rank_match=mr['pinv'][k]['rank']==mh['pinv'][k]['rank'])
            Dr=engine.update_directions(er,cfg['m'],cfg['head_lr']);Dh=engine.update_directions(eh,cfg['m'],cfg['head_lr'])
            dn=norm(Dr);scale=max(1,norm(theta));derr=norm(tuple(x-y for x,y in zip(Dh,Dr)))
            unit=tuple(x/dn for x in Dr)
            slope_ref=-sum(float(np.sum(er[k]*u)) for k,u in zip(('gP','gV','ga'),unit))
            slope_h=-sum(float(np.sum(eh[k]*u)) for k,u in zip(('gP','gV','ga'),unit))
            comparisons=dict(loss_abs=float(abs(eh['L']-er['L'])),weighted_direction_relative=derr/max(dn,1e-300),
                common_slope_abs=abs(slope_h-slope_ref),agop_F_relative=frel,agop_spectral_error=err2,reference_rank_gap=gap,
                agop_error_over_gap=err2/gap if gap>0 else None,Amin_abs=abs(mh['agop_Amin']-mr['agop_Amin']),
                Amean_abs=abs(mh['agop_A']-mr['agop_A']),projector_spectral_difference=projector_difference,refit=refit)
            passes=dict(loss=bool(comparisons['loss_abs']<=1e-7*T.V),
                direction=bool(derr<=1e-4*dn+1e-10*cfg['m']*T.V/scale),
                common_slope=bool(abs(slope_h-slope_ref)<=1e-4*abs(slope_ref)+1e-10*T.V/scale),
                scalar_energy=item['evaluations']['candidate']['scalar_energy_valid'],
                AGOP_matrix=bool(frel<=1e-6),AGOP_gap=gap_condition,
                AGOP_resolution=bool(mr['agop_full_rank_resolved'] and mh['agop_full_rank_resolved']),
                AGOP_alignment_agreement=bool(comparisons['Amin_abs']<=1e-4 and comparisons['Amean_abs']<=1e-4),
                refit_both_stable=bool(point_screen(mr)['refit_stable'] and point_screen(mh)['refit_stable']),
                refit_rank_agreement=bool(all(x['rank_match'] for x in refit.values())),
                refit_cross_risk=bool(all(max(x['self_difference'],x['same_head_risk_error'],x['reverse_same_head_risk_error'])<=1e-7 for x in refit.values())))
            item.update(status='complete',comparisons=comparisons,passes=passes,all_screens_pass=bool(all(passes.values())));write()
        report['all_four_states_pass']=bool(all(x['all_screens_pass'] for x in report['states']))
        report['status']='complete'
    except Exception as exc:report.update(status='incomplete_or_failed',error=repr(exc))
    finally:
        report['inputs_unchanged']=all(sha(HERE/n)==h for n,h in plan['input_sha256'].items());write()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
