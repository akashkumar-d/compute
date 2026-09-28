"""Synthetic/mock tests only. Never import the scientific diagnostic backend."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import planner
import complete_diagnostics as runner


def fixture():
    losses=[1.,.998,.996,.990,.98,.96,.8]
    raw=dict(cfg=dict(seed=7,out='unused',verbose=True),
        Lhist=[[float(i),v] for i,v in enumerate(losses)],
        rows=[dict(step=i,t=float(i),L=v) for i,v in enumerate(losses)],
        termination=dict(step=6,reason='loss_stop'))
    diagnostics=[dict(index=i,step=i,t=float(i),status='ok' if i in (0,4) else 'not_evaluated_runtime_budget',
        original_L=v,metrics={'kept':i} if i in (0,4) else None) for i,v in enumerate(losses)]
    prefixes={str(r):planner.ratio_prefix(raw,r) for r in (1.01,1.05)}
    result=dict(tag='synthetic',cfg=dict(seed=7),diagnostics=diagnostics,
        source_files_sha256={'fake':'123'},source_manifest_sha256='fake',termination=raw['termination'],
        ratio_prefixes=prefixes)
    spec=dict(tag='synthetic',cfg=result['cfg'],source_files_sha256=result['source_files_sha256'],
        source_manifest_sha256='fake',termination=raw['termination'],expected_checkpoint_count=7,
        expected_success_indices=[0,4],original_ratio_prefixes=prefixes)
    return raw,result,dict(tag='synthetic',rows=copy.deepcopy(diagnostics)),spec


class CompletionTests(unittest.TestCase):
    def test_original_prefix_priority_and_verbatim_reuse(self):
        raw,result,partial,spec=fixture();before=copy.deepcopy((raw,result,partial,spec))
        computed,plan=planner.make_plan(raw,result,partial,spec)
        self.assertEqual(plan['initial_1pct_missing_indices'],[1,2])
        self.assertEqual(plan['additional_initial_5pct_missing_indices'],[3,5])
        self.assertEqual(plan['remaining_missing_indices'],[6])
        seen=[];saved=[]
        def evaluate(i):seen.append(i);return {'artificial_metric':i/10}
        rows,origins,attempted=planner.complete_rows(computed,plan,evaluate,
            lambda r,o,a:saved.append(copy.deepcopy((r,o,a))),lambda:False)
        self.assertEqual(seen,[1,2,3,5,6]);self.assertEqual(attempted,seen)
        self.assertEqual(len(saved),6)
        for i in [0,4]:
            self.assertEqual(rows[i],before[1]['diagnostics'][i])
            self.assertEqual(origins[str(i)],'original_interrupted_diagnostic_process')
        self.assertEqual((raw,result,partial,spec),before)
        self.assertTrue(all(r['status']=='ok' for r in rows))

    def test_hash_mismatch_and_path_escape_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);p=root/'input.json';p.write_text('{}')
            spec={'files':{'input.json':dict(size=2,sha256=planner.sha256(p))}}
            self.assertEqual(planner.verify_inputs(spec,root),spec['files'])
            p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'Pinned input changed'):planner.verify_inputs(spec,root)
            with self.assertRaisesRegex(ValueError,'plain file'):planner.verify_inputs({'files':{'../out':{}}},root)
            q=root/'symlink';q.symlink_to(Path(__file__).resolve())
            with self.assertRaisesRegex(ValueError,'escaped'):planner.verify_inputs({'files':{'symlink':{}}},root)

    def test_mismatches_and_new_success_refused(self):
        for kind in ('cfg','kernel','manifest','termination','grid','clock','partial','prefix','success'):
            raw,result,partial,spec=fixture()
            if kind=='cfg':raw['cfg']['seed']=8
            elif kind=='kernel':result['source_files_sha256']={'changed':'x'}
            elif kind=='manifest':result['source_manifest_sha256']='changed'
            elif kind=='termination':raw['termination']=dict(step=5)
            elif kind=='grid':result['diagnostics'][1]['step']=2;partial['rows']=copy.deepcopy(result['diagnostics'])
            elif kind=='clock':raw['Lhist'][1][0]=8.
            elif kind=='partial':partial['rows'][0]['metrics']={'changed':1}
            elif kind=='prefix':spec['original_ratio_prefixes']=dict(bad=True)
            else:result['diagnostics'][1]['status']='ok';partial['rows']=copy.deepcopy(result['diagnostics'])
            with self.assertRaises(ValueError,msg=kind):planner.make_plan(raw,result,partial,spec)

    def test_budget_preserves_unattempted_states_and_prior_success(self):
        raw,result,partial,spec=fixture();computed,plan=planner.make_plan(raw,result,partial,spec)
        seen=[];snapshots=[]
        def evaluate(i):seen.append(i);return {'fake':i}
        rows,origins,attempted=planner.complete_rows(computed,plan,evaluate,
            lambda r,o,a:snapshots.append(copy.deepcopy(r)),lambda:len(seen)>=2)
        self.assertEqual(seen,[1,2]);self.assertEqual(rows[3],result['diagnostics'][3])
        self.assertEqual(rows[4],result['diagnostics'][4]);self.assertEqual(len(snapshots),3)

    def test_failed_state_retained_and_rest_attempted(self):
        raw,result,partial,spec=fixture();computed,plan=planner.make_plan(raw,result,partial,spec)
        def evaluate(i):
            if i==2:raise ArithmeticError('mock failure')
            return {'fake':i}
        rows,origins,attempted=planner.complete_rows(computed,plan,evaluate,lambda *args:None,lambda:False)
        self.assertEqual(attempted,[1,2,3,5,6]);self.assertEqual(rows[2]['status'],'failed')
        self.assertIn('mock failure',rows[2]['error']);self.assertEqual(rows[6]['status'],'ok')

    def test_metrics_cannot_replace_identity(self):
        raw,result,partial,spec=fixture();computed,plan=planner.make_plan(raw,result,partial,spec)
        rows,_,_=planner.complete_rows(computed,plan,lambda i:{'step':999},lambda *args:None,lambda:False)
        self.assertEqual(rows[1]['step'],1);self.assertEqual(rows[1]['status'],'failed')

    def test_runner_refuses_local_before_input_or_scientific_import(self):
        with mock.patch.object(runner.launch.sys,'platform','darwin'),mock.patch.object(
                runner,'verify_inputs',side_effect=AssertionError('Inputs must not be reached')):
            with self.assertRaisesRegex(RuntimeError,'Execution requires'):
                runner.main(['--config','unused','--tag','fake','--out-dir','unused',
                    '--diagnostic-seconds','1200','--deadline-utc','2026-09-29T00:00:00Z'])

    def test_mock_backend_end_to_end_and_input_hash_guard(self):
        # Arrays are small synthetic fixtures; real diagnostic/engine modules are never imported.
        import numpy as np
        import types, os, sys, time
        for tamper in (False,True):
            with self.subTest(tamper=tamper),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
                root=Path(td).resolve();source=root/'original';source.mkdir();code=root/'swiglu/code';code.mkdir(parents=True)
                (root/'configs').mkdir();output=root/'execution/data/synthetic';output.parent.mkdir(parents=True)
                raw,result,partial,spec=fixture()
                cfg=dict(seed=7,d=2,m=2);raw['cfg']=dict(cfg,out='unused',verbose=True);result['cfg']=cfg;spec['cfg']=cfg
                (code/'fake.py').write_text('# pinned mock kernel; never imported\n')
                hashes={'fake.py':planner.sha256(code/'fake.py')}
                (code/'SOURCE_MANIFEST.json').write_text(json.dumps({'status':'frozen_after_review'}))
                spec['source_files_sha256']=result['source_files_sha256']=hashes
                spec['source_manifest_sha256']=result['source_manifest_sha256']=planner.sha256(code/'SOURCE_MANIFEST.json')
                result.update(completed=True,cell='mock',rank=1,all_update_count=7)
                for name,value in [('raw.json',raw),('result.json',result),('diagnostics_partial.json',partial)]:
                    (source/name).write_text(json.dumps(value))
                np.savez(source/'snaps.npz',t=np.arange(7,dtype=float),P=np.ones((7,3,2)),V=np.ones((7,3,2))*2,a=np.ones((7,2)))
                spec.update(remote_input_root=str(source),raw_name='raw.json',snapshots_name='snaps.npz',files={
                    p.name:dict(size=p.stat().st_size,sha256=planner.sha256(p)) for p in source.iterdir()})
                (root/'INPUT_SPEC.json').write_text(json.dumps(spec))
                _,plan=planner.make_plan(raw,result,partial,spec)
                (root/'COMPLETION_PLAN.json').write_text(json.dumps(plan))
                cfgpath=root/'configs/synthetic.json';cfgpath.write_text(json.dumps([{'tag':'synthetic','args':cfg}]))
                manifest=dict(runtime={},source_sha256=hashes,configs=[dict(id='synthetic',config_path='configs/synthetic.json')])
                fake=types.ModuleType('diagnostics');fake.DIAGNOSTIC_SCHEMA_VERSION='mock';fake.AGOP_RESOLUTION_GUARD=1e-9
                fake.make_engine=mock.Mock(return_value='fake_engine')
                def metrics(engine,p,v,a):
                    self.assertEqual(engine,'fake_engine');self.assertFalse(p.flags.writeable)
                    self.assertFalse(v.flags.writeable);self.assertFalse(a.flags.writeable)
                    return {'synthetic_metric':float(p.sum()+v.sum()+a.sum())}
                fake.metrics=mock.Mock(side_effect=metrics)
                before={p.name:p.read_bytes() for p in source.iterdir()}
                if tamper:(source/'raw.json').write_text('{}')
                with mock.patch.object(runner,'ROOT',root),mock.patch.object(runner,'CODE',code), \
                     mock.patch.object(runner.launch,'require_server'),mock.patch.object(runner.launch,'validate',return_value=(manifest,'mockhash')), \
                     mock.patch.object(runner.launch,'validate_protocol'),mock.patch.object(runner.launch,'require_review'), \
                     mock.patch.object(runner.launch,'require_capacity'),mock.patch.object(runner.launch,'require_priority'), \
                     mock.patch.object(runner.os,'nice',return_value=10),mock.patch.dict(sys.modules,{'diagnostics':fake}), \
                     mock.patch.dict(os.environ,{**{k:'1' for k in runner.launch.THREAD_VARIABLES},
                         'CUDA_VISIBLE_DEVICES':'','AGOP_DIAGNOSTIC_LAUNCHER_PID':str(os.getppid())}):
                    args=['--config',str(cfgpath),'--tag','synthetic','--out-dir',str(output),
                          '--diagnostic-seconds','1200','--deadline-utc',runner.launch.utc(time.time()+1400)]
                    if tamper:
                        with self.assertRaisesRegex(ValueError,'Pinned input changed'):runner.main(args)
                        fake.make_engine.assert_not_called();fake.metrics.assert_not_called();self.assertFalse(output.exists())
                    else:
                        self.assertEqual(runner.main(args),0)
                        receipt=json.loads((output/'DIAGNOSTICS_COMPLETION.json').read_text())
                        self.assertTrue(receipt['completion_pass_complete']);self.assertTrue(receipt['input_files_unchanged_after'])
                        self.assertFalse(receipt['original_process_diagnostics_complete']);self.assertEqual(receipt['new_training_updates'],0)
                        self.assertEqual(fake.metrics.call_count,5)
                        for i in [0,4]:self.assertEqual(receipt['diagnostics'][i],result['diagnostics'][i])
                        self.assertEqual({p.name:p.read_bytes() for p in source.iterdir()},before)
                        self.assertFalse((output/'result.json').exists())

    def test_no_training_backend_calls_in_adapter(self):
        import ast
        tree=ast.parse(Path(runner.__file__).read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        backend=[x for x in calls if x.startswith('diagnostics.')]
        self.assertEqual(sorted(backend),['diagnostics.make_engine','diagnostics.metrics'])
        self.assertFalse(any(x in calls for x in ('engine.run','diagnostics.run','run_one.main')))

if __name__=='__main__':unittest.main()
