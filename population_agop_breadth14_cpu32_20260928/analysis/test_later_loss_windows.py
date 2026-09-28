import copy
import math
import random
import unittest
from later_loss_windows import longest, select, evaluate

class SecondaryWindows(unittest.TestCase):
    def test_longest_against_exhaustive_enumeration(self):
        rng=random.Random(73028)
        for _ in range(500):
            n=rng.randint(1,25);loss=[rng.choice([float('nan'),0,.2,.95,.99,1,1.03,1.1,2]) for _ in range(n)]
            clock=[0.]
            for i in range(1,n):clock.append(clock[-1]+rng.choice([0.,.1,1.,2.]))
            for ratio in (1.01,1.05):
                band=(.5,1.2);choices=[]
                for a in range(n):
                    for b in range(a,n):
                        xs=loss[a:b+1]
                        if all(math.isfinite(x) and 0<x and band[0]<=x<=band[1] for x in xs) and max(xs)/min(xs)<=ratio:
                            choices.append((clock[b]-clock[a],-a,-b))
                expected=max(choices) if choices else None
                actual,_=longest(loss,clock,ratio,band)
                self.assertEqual(actual,None if expected is None else (-expected[1],-expected[2]))

    def fixture(self):
        point=lambda i,A,R:dict(step=i,time=float(i),loss_raw=1.,A_min=A,refit_low_raw=R,refit_high_raw=R,
            diagnostic_status='ok',issues=[],agop_screen=True,refit_screen=True,raw_refit_nonnegative_screen=True)
        arm=dict(id='a',teacher='h2',r=8,target_variance=1.,issues=[],snapshot_integrity_verified=True,
            history=[dict(step=i,time=float(i),loss_raw=1. if i<=100 else .8) for i in range(102)],
            checkpoints=[point(0,0.,1.),point(50,.6,.7),point(100,.7,.6)])
        return dict(snapshot_usable=True,arms=[arm]),dict(rank=8,rows=[dict(teacher='h2',target_variance=1.,nonlinear_energy=1.)])

    def test_selection_is_feature_independent(self):
        s,e=self.fixture();a=select(s,e);s['arms'][0]['checkpoints']=[dict(unrelated='altered')]
        self.assertEqual(select(s,e),a)

    def test_fixed_endpoints_and_later_drop(self):
        s,e=self.fixture();out=evaluate(select(s,e),s)
        self.assertTrue(all(w['joint_candidate'] and w['later_loss_release'] for w in out['runs'][0]['windows']))
        s['arms'][0]['checkpoints'][0]['diagnostic_status']='not_evaluated'
        out=evaluate(select(s,e),s)
        self.assertTrue(all(not w['joint_candidate'] and w['endpoint_steps']==[0,100] for w in out['runs'][0]['windows']))

    def test_unknown_snapshot_cannot_qualify(self):
        s,e=self.fixture();s['arms'][0]['snapshot_integrity_verified']=False
        self.assertFalse(any(w['joint_candidate'] for w in evaluate(select(s,e),s)['runs'][0]['windows']))

    def test_changed_snapshot_rejected_and_no_history_retained(self):
        s,e=self.fixture();s['snapshot_usable']=False
        with self.assertRaises(ValueError):select(s,e)
        s['snapshot_usable']=True;s['arms'][0]['history']=[]
        out=select(s,e);self.assertEqual(len(out['runs']),1);self.assertEqual(out['runs'][0]['windows'],[])

if __name__=='__main__':unittest.main()
