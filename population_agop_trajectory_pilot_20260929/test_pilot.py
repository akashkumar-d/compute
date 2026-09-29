"""Pure-array controller/contracts and synthetic scalar polynomial tests only.

No population/teacher/student model import or evaluation is allowed here.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
import controller as c
import comparisons as checks
import runtime
import run as entry
import hermite_audit_only as high

HERE=Path(__file__).resolve().parent
SETTINGS=json.loads((HERE/'SETTINGS.json').read_text())
CFG=dict(d=2,m=2,seed=9,s=.3,head_ratio=1.,head_lr=.01,h=.01,dt_max=50.,t_max=3000.,
         cp_ratio=1.06,cp_min=.5,dl_ratio=1.5,L_stop=.01,max_steps=2,c=[1.])


def theta(): return (np.ones((3,2)),np.ones((3,2)),np.ones(2))
def ev(x):
    return dict(L=sum(float(np.sum(z*z))/2 for z in x),V=10.,G=np.eye(2),t=np.zeros(2),
                gP=x[0].copy(),gV=x[1].copy(),ga=x[2].copy(),Cp={'x':np.zeros((2,2))},D={'x':np.zeros(2)})
def metric(resolved=True,stable=True,Amin=0.,refit=.9):
    return dict(status='ok',agop_full_rank_resolved=resolved,agop_Amin=Amin,agop_A=Amin,
        equilibrated_lambda_ratio=1. if stable else -1.,
        pinv={str(x):dict(actual_mse=refit,relative_normal_residual=0.,rank=2) for x in checks.CUTOFFS})


class ControllerTests(unittest.TestCase):
    def test_accept_and_no_mutation(self):
        x=theta();old=copy.deepcopy(x);logs=[]
        y,new,dt,rel=c.backtrack(x,ev(x),CFG,0.,SETTINGS['controller'],lambda y,j:ev(y),logs.append)
        self.assertLess(new['L'],ev(x)['L']);self.assertEqual(logs[-1]['status'],'accepted')
        for a,b in zip(x,old): np.testing.assert_array_equal(a,b)
    def test_backtrack_overshoot_and_exhaustion(self):
        cfg=dict(CFG,h=100,dt_max=10)
        logs=[];x=theta()
        c.backtrack(x,ev(x),cfg,0.,SETTINGS['controller'],lambda y,j:ev(y),logs.append)
        self.assertTrue(any(r['status']=='rejected' for r in logs));self.assertEqual(logs[-1]['status'],'accepted')
        with self.assertRaises(c.NumericalFailure):
            c.backtrack(x,ev(x),cfg,0.,SETTINGS['controller'],lambda y,j:dict(ev(y),L=99.),lambda r:None)
    def test_stall_and_nonfinite_are_never_accepted(self):
        x=theta();e=ev(x)
        for t in (CFG['t_max'],):
            with self.assertRaises(c.NumericalStall): c.backtrack(x,e,CFG,t,SETTINGS['controller'],lambda *_:self.fail(),lambda r:None)
        with self.assertRaises(c.NumericalFailure):
            c.backtrack(x,e,CFG,0,SETTINGS['controller'],lambda y,j:dict(ev(y),L=float('nan')),lambda r:None)
    def test_pairing_tiny_synthetic_initials(self):
        for seed in (9,10):
            a,ra=c.initialize(dict(CFG,seed=seed));b,rb=c.initialize(dict(CFG,seed=seed,s=1.,head_ratio=.3))
            np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(ra['gaussian_hashes'],rb['gaussian_hashes'])
            np.testing.assert_allclose(a[0],.3*b[0],rtol=3e-16,atol=1e-17)
    def test_exact_window_boundary_and_same_checkpoint(self):
        criteria=dict(initial_loss_ratio_tolerances=[1.01,1.05],delta_Amin_min=.5,
            refit_gain_over_variance_min=.1,later_raw_loss_drop_over_variance_min=.1)
        events=c.Events(criteria)
        events.observe(0,0.,1.,metric(),checks.point_screen(metric()))
        events.observe(1,1.,1/1.01,metric(Amin=.6,refit=.9),checks.point_screen(metric()))
        self.assertEqual(events.exits,{})
        events.observe(2,2.,.991,metric(Amin=.2,refit=.6),checks.point_screen(metric()))
        self.assertEqual(events.candidates,{})
        events.observe(3,3.,.991,metric(Amin=.6,refit=.6),checks.point_screen(metric()))
        self.assertEqual(set(events.candidates),{'1.01','1.05'})
        events.observe(4,4.,.8,metric(),checks.point_screen(metric()))
        self.assertEqual(set(events.releases),{'1.01','1.05'})
        self.assertEqual(events.exits['1.01']['last_valid_step'],3)
    def test_unresolved_initial_cannot_qualify(self):
        criteria=dict(initial_loss_ratio_tolerances=[1.01],delta_Amin_min=.5,refit_gain_over_variance_min=.1,later_raw_loss_drop_over_variance_min=.1)
        e=c.Events(criteria);m=metric(resolved=False)
        e.observe(0,0,1,m,checks.point_screen(m));e.observe(1,1,1,metric(Amin=.8,refit=.1),checks.point_screen(metric()))
        self.assertEqual(e.candidates,{})
    def test_event_checkpoint_does_not_advance_natural_grid(self):
        q=c.Checkpoints(CFG);self.assertTrue(q.natural(0,1));next_cp=q.next_cp
        self.assertFalse(q.natural(.1,1));self.assertEqual(q.next_cp,next_cp)


class ComparisonTests(unittest.TestCase):
    def records(self,resolved=True,stable=True):
        x=theta();return x,(ev(x),np.diag([3.,1.]),metric(resolved,stable))
    def test_agreement_unresolved_vs_disagreement(self):
        x,a=self.records(False,False);_,b=self.records(False,False)
        z=checks.compare(a,b,x,CFG,SETTINGS);self.assertTrue(z['passed']);self.assertIsNone(z['passes']['AGOP_gap'])
        _,b=self.records(True,False);self.assertFalse(checks.compare(a,b,x,CFG,SETTINGS)['passed'])
        _,b=self.records(False,True);self.assertFalse(checks.compare(a,b,x,CFG,SETTINGS)['passed'])
    def test_failure_gates(self):
        x,a=self.records();b=copy.deepcopy(a);b[0]['L']+=1e-3
        self.assertFalse(checks.compare(a,b,x,CFG,SETTINGS)['passes']['loss'])
        for value in (float('nan'),-1e-7):
            e=ev(x);e['L']=value
            with self.assertRaises(c.NumericalFailure): checks.validate_ev(e)
        e=ev(x);e['L']=-1e-11;checks.validate_ev(e)
        e['hermite_diagnostics']=dict(all_scalar_estimates_valid=False,scalar_bank=dict(material_negative_energy=np.array([True])))
        with self.assertRaises(c.NumericalFailure): checks.validate_ev(e)
    def test_refit_cross_orientation(self):
        x,a=self.records();b=copy.deepcopy(a);b[0]['t']=np.array([.1,.2])
        z=checks.compare(a,b,x,CFG,SETTINGS)
        self.assertFalse(z['passes']['refit_cross_risk'])
        self.assertGreater(z['refit']['1e-08']['candidate_on_reference'],z['refit']['1e-08']['reference_self'])


class RuntimeMockTests(unittest.TestCase):
    def test_cached_evaluation_never_calls_engine(self):
        E=mock.Mock();x=theta();e=ev(x);cached=runtime.Cached(E,x,e)
        self.assertIs(cached.evaluate(*x),e);E.evaluate.assert_not_called()
        with self.assertRaises(ValueError): cached.evaluate(x[0]+1,x[1],x[2])
    def test_deadline_prevents_call(self):
        with tempfile.TemporaryDirectory() as d:
            S=runtime.Session(HERE,Path(d),SETTINGS,time.monotonic()-200,[False],dict(mode='smoke',model_evaluations=0))
            E=mock.Mock()
            with self.assertRaises(runtime.BudgetStop): S.evaluate(E,theta(),'mock')
            E.evaluate.assert_not_called()
    def test_atomic_commit_and_initial_reference_before_update(self):
        class FakeE:
            T=type('T',(),{'V':10.})()
            base='base';high='high';reference='reference'
        with tempfile.TemporaryDirectory() as d:
            S=runtime.Session(HERE,Path(d),SETTINGS,time.monotonic(),[False],dict(mode='arm',model_evaluations=0))
            order=[];saved=[]
            S.evaluate=lambda E,x,label:(order.append(label) or ev(x))
            S.audit=lambda *args,**kw:order.append('fullref' if kw.get('full_reference') else 'high')
            S.snapshot=lambda label,x,rec,step,t,roles:saved.append((c.state_hash(x),float(rec[0]['L']),step))
            calls=[0]
            def diag(E,x,e):
                calls[0]+=1
                if calls[0]>1: raise c.NumericalFailure('artificial diagnostic failure')
                return e,np.eye(2),metric()
            criteria=dict(initial_loss_ratio_tolerances=[1.01,1.05],delta_Amin_min=.5,
                refit_gain_over_variance_min=.1,later_raw_loss_drop_over_variance_min=.1)
            with mock.patch.object(runtime,'Evaluators',return_value=FakeE()),mock.patch.object(runtime,'diagnostic',side_effect=diag):
                runtime.run_arm(S,dict(id='tiny',configuration=CFG),criteria)
            self.assertEqual(order[:2],['initial','fullref']);self.assertEqual(S.report['accepted_updates'],0)
            self.assertEqual(S.report['status'],'failed');self.assertTrue((Path(d)/'failure_uncommitted_step1.npz').exists())
            self.assertEqual(len({x[0] for x in saved}),1)
    def test_server_gate_and_review_refusal_no_models(self):
        import probe_resources
        with mock.patch.object(sys,'platform','darwin'):
            with self.assertRaises(RuntimeError): probe_resources.require_server()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'REVIEW.json').write_text('{"status":"approved_for_execution","manifest_sha256":"other","reviewer":"x"}')
            with self.assertRaises(RuntimeError): entry.require_review(p,'expected')
        self.assertFalse(any(n in sys.modules for n in ('swpop','swsmall','student_transition','student_transition_fast','teacher_h3','diagnostics')))


class AuditScalarTests(unittest.TestCase):
    def polynomial(self,x,max_order=4):
        return [x**3,3*x*x,6*x,np.full_like(x,6.),np.zeros_like(x)][:max_order+1]
    def test_256_32_polynomial_coefficients_and_zero(self):
        b=np.array([.2,-.3,0.]);s=np.array([.6,1.2,0.])
        bank=high.scalar_bank(b,s,derivative_fn=self.polynomial)
        coef=bank['coefficients'];exact=np.zeros_like(coef)
        exact[:,0,0]=b**3+3*b*s*s;exact[:,0,1]=3*b*b*s+3*s**3
        exact[:,0,2]=3*b*s*s*math.sqrt(2);exact[:,0,3]=s**3*math.sqrt(6)
        exact[:,1,0]=3*(b*b+s*s);exact[:,1,1]=6*b*s;exact[:,1,2]=3*s*s*math.sqrt(2)
        exact[:,2,0]=6*b;exact[:,2,1]=6*s;exact[:,3,0]=6
        np.testing.assert_allclose(coef[:,:,:4],exact[:,:,:4],rtol=3e-12,atol=5e-13)
        self.assertLess(float(np.max(np.abs(coef[:,:,4:]))),5e-10)
        self.assertTrue(bank['all_estimates_valid']);self.assertEqual(bank['node_counts'][2],0)
    def test_negative_energy_preserved(self):
        def bad(*a,**kw):return np.array([0.]),np.array([2.])
        def const(x,max_order=4):return [np.ones_like(x)]+[np.zeros_like(x)]*4
        bank=high.scalar_bank([0.],[1.],derivative_fn=const,rule_fn=bad)
        self.assertFalse(bank['all_estimates_valid']);self.assertLess(bank['signed_residual_energy'][0,0],0)
        self.assertTrue(np.isnan(bank['residual_energy_estimate'][0,0]))
    def test_exact_allowed_source_diff(self):
        base=(HERE/'hermite_pairs.py').read_text();expected=base.replace(
            '"""Diagnostic-only scalar Hermite and Gaussian pair moments; no model imports.',
            '"""Audit-only N256/q32 clone of reviewed Hermite moments; never the training evaluator.')
        expected=expected.replace('degree=64, order=16,','degree=256, order=32,').replace('(64,128)','(256,)').replace('(16,24)','(32,)')
        expected=expected.replace('hermite_degree=64,hermite_quad_order=16','hermite_degree=256,hermite_quad_order=32')
        self.assertEqual(expected,(HERE/'hermite_audit_only.py').read_text())
        for name,h in json.loads((HERE/'FROZEN_SOURCES.json').read_text())['source_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),h)

if __name__=='__main__': unittest.main()
