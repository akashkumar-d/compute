"""Meaningful analytic invariants and cancellation regressions; no training."""
import unittest
import numpy as np
from stable_relu import population_agop,analytic_jacobian,_joint_negative,fixed_checkpoint_svd


class StableDiagnosticsTest(unittest.TestCase):
    def test_independent_and_degenerate_gate_probabilities(self):
        from scipy.special import ndtr
        for a,b in [(-.2,-.7),(-9.,-10.),(0.,0.)]:
            v,_,_=_joint_negative(a,b,0.,1.,1e-12)
            self.assertAlmostEqual(v/(ndtr(a)*ndtr(b)),1.,places=12)
            v,_,_=_joint_negative(a,b,1.,0.,1e-12)
            self.assertAlmostEqual(v/ndtr(min(a,b)),1.,places=12)
            self.assertEqual(_joint_negative(a,b,-1.,0.,1e-12)[0],0.)

    def test_zero_bias_orthogonal_known_kernel(self):
        W=np.eye(4);A=np.array([1.,-2.,3.,-.5]);b=np.zeros(4);U=np.eye(4)[:,:2]
        G,info,_=population_agop(A,W,b,U,pair_tolerance=1e-12)
        Q=.25*np.ones((4,4))+.25*np.eye(4)
        expected=W.T@(Q*np.outer(A,A))@W/16
        np.testing.assert_allclose(G,expected,rtol=1e-13,atol=1e-15)
        self.assertEqual(info['quadrature_warning_count'],0)

    def test_saturated_cancellation_is_rank_one(self):
        rng=np.random.default_rng(91);W=rng.normal(size=(8,5));A=rng.normal(size=8)
        # Cancellation can make a direct signed Q contraction indefinite.
        W[7]=-(A[:7,None]*W[:7]).sum(axis=0)/A[7]+1e-12
        b=np.full(8,1000.);U=np.eye(5)[:,:2]
        G,info,_=population_agop(A,W,b,U)
        grad=(A.astype(np.longdouble)[:,None]*W.astype(np.longdouble)).sum(axis=0)/8
        expected=np.asarray(np.outer(grad,grad),float)
        np.testing.assert_allclose(G,expected,rtol=1e-14,atol=0.)
        self.assertFalse(info['numerical_screen_passed'])
        self.assertEqual(info['numerical_rank'],1)
        self.assertEqual(info['underflowed_rare_probability_count'],8)

    def test_leaky_linear_gradient_and_directional_derivative(self):
        rng=np.random.default_rng(17);W=rng.normal(size=(6,4));A=rng.normal(size=6);b=rng.normal(size=6)
        X=rng.normal(size=(50,4));D=rng.normal(size=(50,4));alpha=.2;h=1e-6
        def pred(X):
            Z=X@W.T+b
            return (alpha*Z+(1-alpha)*np.maximum(Z,0))@A/6
        crossing=((X+h*D)@W.T+b>0)!=((X-h*D)@W.T+b>0)
        keep=~np.any(crossing,axis=1)
        fd=(pred(X+h*D)-pred(X-h*D))/(2*h)
        actual=np.sum(analytic_jacobian(A,W,b,X,alpha)*D,axis=1)
        np.testing.assert_allclose(fd[keep],actual[keep],rtol=1e-7,atol=1e-9)

    def test_gate_covariance_against_large_independent_sample(self):
        rng=np.random.default_rng(100);W=rng.normal(size=(4,4));A=np.array([1.,-.5,.75,-.2]);b=np.array([-.5,.8,-.2,1.2]);U=np.eye(4)[:,:2]
        G,info,_=population_agop(A,W,b,U,alpha=.1)
        X=rng.normal(size=(30000,4));J=analytic_jacobian(A,W,b,X,alpha=.1)
        empirical=J.T@J/len(X)
        self.assertLess(np.linalg.norm(G-empirical)/np.linalg.norm(G),.03)
        self.assertEqual(info['quadrature_warning_count'],0)

    def test_fast_formula_against_independent_adaptive_route(self):
        rng=np.random.default_rng(319);W=rng.normal(size=(7,4));A=rng.normal(size=7);b=rng.normal(size=7);U=np.eye(4)[:,:2]
        fast,_,_=population_agop(A,W,b,U,alpha=.2)
        adaptive,_,_=population_agop(A,W,b,U,alpha=.2,pair_backend='adaptive',pair_tolerance=1e-12)
        np.testing.assert_allclose(fast,adaptive,rtol=2e-10,atol=2e-13)

    def test_rare_tail_fallback_and_zero_spatial_rows(self):
        W=np.array([[1.,0.,0.],[.5,np.sqrt(.75),0.],[0.,0.,0.]])
        A=np.array([1.,-.8,.4]);b=np.array([-9.,-10.,2.]);U=np.eye(3)[:,:1]
        fast,info,_=population_agop(A,W,b,U)
        adaptive,_,_=population_agop(A,W,b,U,pair_backend='adaptive')
        np.testing.assert_allclose(fast,adaptive,rtol=1e-9,atol=1e-40)
        self.assertEqual(info['constant_spatial_row_count'],1)
        import json
        json.dumps(info,allow_nan=False)


if __name__=='__main__':
    unittest.main()
