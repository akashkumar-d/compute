"""Four audit-delta checks; artificial arrays only, no scientific module imports."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import audit

ROOT=audit.ROOT
class DeltaTests(unittest.TestCase):
    def test_actual_weighted_cap_and_cached_single_evaluation(self):
        source=ast.parse((ROOT/'swiglu/code/engine/swsmall_periodic.py').read_text())
        nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_head_lr','update_directions')]
        scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure update math>','exec'),scope)
        P=np.array([[1.,2.],[3.,4.]]);V=P/2;a=np.array([.2,.1]);ev=dict(gP=P/4,gV=V/3,ga=np.array([100.,200.]),L=.7,V=1.)
        calls={'evaluate':0,'agop':0}
        class Fake:
            T=types.SimpleNamespace(r=1)
            def evaluate(self,*args,**kwargs):calls['evaluate']+=1;return ev
            def agop(self,*args):calls['agop']+=1;return np.diag([1.,2.])
            def refit(self,value):return .2
        def metrics(E,p,v,w):
            self.assertIs(E.evaluate(p,v,w,need_grad=True),ev)
            self.assertEqual(E.agop(p,v,w,ev).shape,(2,2));self.assertEqual(E.refit(ev),.2)
            with self.assertRaises(ValueError):E.evaluate(p.copy(),v,w)
            return {'L':.7}
        cfg=dict(m=2,head_lr=.01,h=.01,dt_max=50.,c=[1.],n_pair=32,n_diag=96,n_z=48,n_x=24)
        row,cache=audit.order_record(np,types.SimpleNamespace(metrics=metrics),types.SimpleNamespace(update_directions=scope['update_directions']),Fake(),cfg,1,P,V,a)
        expected=[2*ev['gP'],2*ev['gV'],.02*ev['ga']]
        theta=np.sqrt((P**2).sum(0)+(V**2).sum(0)+a*a)
        rate=np.sqrt(sum(x*x for x in expected[:2]).sum(0)+expected[2]**2)/theta
        self.assertEqual(calls,{'evaluate':1,'agop':1})
        np.testing.assert_array_equal(cache['direction'],np.concatenate([x.ravel() for x in expected]))
        self.assertEqual(row['dt'],.01/rate.max());self.assertEqual(row['cap_neuron'],int(rate.argmax()))

    def test_same_order_initialization_controls_threshold(self):
        canonical=audit.module(ROOT/'reference/canonical_summary.py','mock_saved_criteria')
        def metrics(A,risk):
            return dict(L=1.,agop_Amin=A,agop_A=A,agop_Atop=A,pinv={str(c):dict(actual_mse=risk,relative_normal_residual=0.) for c in [1e-8,1e-10,1e-12,1e-14]},agop_full_rank_resolved=True,agop_psd_resolved=True,agop_resolution_guard=1e-9,agop_rank_r_eigenvalue_relative=.5,agop_rank_r_relative_gap=.1,equilibrated_lambda_ratio=.2)
        rows=[dict(step=0,time=0.,orders={str(f):dict(normalization_variance=1.,metrics=metrics(A,.8)) for f,A in [(1,.1),(2,.3)]}),dict(step=10,time=1.,orders={str(f):dict(normalization_variance=1.,metrics=metrics(.7,.3)) for f in [1,2]})]
        audit.qualify(canonical,{},rows)
        self.assertTrue(rows[1]['orders']['1']['fixed_state_qualification']['same_checkpoint_material_candidate_numerically_qualified'])
        self.assertFalse(rows[1]['fixed_candidate_thresholds_survive_doubling'])
        rows[0]['orders']['1']['metrics']['agop_Amin']=.3
        rows[0]['orders']['2']['metrics']['agop_Amin']=.1
        audit.qualify(canonical,{},rows)
        self.assertFalse(rows[1]['native_order_fixed_candidate_qualifies'])
        self.assertTrue(rows[1]['doubled_order_fixed_candidate_qualifies'])
        self.assertFalse(rows[1]['fixed_candidate_thresholds_survive_doubling'])

    def test_runtime_delta_and_local_refusal(self):
        manifest=audit.launch.load_json(ROOT/'MANIFEST.json');audit.launch.validate_protocol(manifest,ROOT)
        for field,bad in [('workers',3),('global_seconds',601),('per_arm_seconds',271)]:
            other=copy.deepcopy(manifest);other['runtime'][field]=bad
            with self.assertRaises(ValueError):audit.launch.validate_protocol(other,ROOT)
        cmd=audit.launch.build_command(ROOT,ROOT/'mock',manifest['configs'][0],manifest['runtime'],270,'2099-01-01T00:00:00+00:00')
        self.assertEqual(cmd[cmd.index('--diagnostic-seconds')+1],'250.0')
        if not __import__('sys').platform.startswith('linux'):
            with self.assertRaises(RuntimeError):audit.launch.require_server()

    def test_all_nine_slots_retained_with_missing_and_failed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'SELECTION.json').write_bytes((ROOT/'SELECTION.json').read_bytes());out=root/'output';out.mkdir()
            case=json.loads((root/'SELECTION.json').read_text())['cases'][0];folder=out/'data'/case['id'];folder.mkdir(parents=True)
            first=dict(case['states'][0],tag=case['id'],status='failed_or_censored',orders={},error='artificial')
            (folder/'FIXED_STATE_AUDIT.json').write_text(json.dumps(dict(rows=[first],completed=False)))
            audit.launch.collect_outcomes(root,out);result=json.loads((out/'ALL_STATES.json').read_text())
            self.assertEqual(len(result['rows']),9);self.assertEqual(result['rows'][0]['error'],'artificial')
            self.assertEqual(sum(r['status']=='not_evaluated_no_child_record' for r in result['rows']),8)
            self.assertFalse(result['all_case_reports_complete'])

if __name__=='__main__':unittest.main()
