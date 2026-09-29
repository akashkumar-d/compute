"""Synthetic polynomial/Gaussian contractions only; no model imports/evaluations."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

# Load only the existing scalar rule; do not import a population engine.
SCALAR=Path(__file__).resolve().parent/'transition_rule.py'
spec=importlib.util.spec_from_file_location('transition_rule',SCALAR)
scalar=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)
sys.modules.setdefault('transition_rule',scalar)
import hermite_pairs as hp


def polynomial(x,max_order=4):
    x=np.asarray(x)
    values=[x**3,3*x*x,6*x,np.full_like(x,6.),np.zeros_like(x)]
    return values[:max_order+1]


def polynomial_bank(bias,scale,degree=64):
    b,s=np.asarray(bias),np.asarray(scale);m=len(b)
    c=np.zeros((m,5,degree+1))
    c[:,0,0]=b**3+3*b*s*s;c[:,0,1]=3*b*b*s+3*s**3
    c[:,0,2]=3*b*s*s*math.sqrt(2);c[:,0,3]=s**3*math.sqrt(6)
    c[:,1,0]=3*(b*b+s*s);c[:,1,1]=6*b*s;c[:,1,2]=3*s*s*math.sqrt(2)
    c[:,2,0]=6*b;c[:,2,1]=6*s;c[:,3,0]=6
    return dict(coefficients=c,residual_energy_estimate=np.zeros((m,5)),degree=degree)


def synthetic_case():
    pi=np.array([[1.,.3,-1.,0.],[.2,.7,-.2,0.],[0.,.4,0.,0.]])
    nu=np.array([[.1,-.4,.3,.2],[.7,.2,-.1,.9],[.5,.3,.4,-.6]])
    p0=np.array([.2,-.3,1.,-.5]);v0=np.array([.4,-.1,.5,.2])
    return pi,p0,nu,v0


def direct_polynomial_fields(pi,p0,nu,v0):
    z,w=np.polynomial.hermite_e.hermegauss(6);w=w/math.sqrt(2*math.pi)
    grids=np.meshgrid(z,z,z,indexing='ij');x=np.stack([a.ravel() for a in grids],axis=1)
    weights=(w[:,None,None]*w[None,:,None]*w[None,None,:]).ravel()
    gates=x@pi+p0;values=x@nu+v0
    s0,s1,s2=polynomial(gates,max_order=2);m=len(p0)
    result={name:np.empty((m,m)) for name in hp.PAIR_KEYS}
    for j in range(m):
        for k in range(m):
            Vj,Vk=values[:,j],values[:,k]
            products=(s0[:,j]*s0[:,k]*Vj*Vk,s2[:,j]*s0[:,k]*Vj*Vk,
                s1[:,j]*s1[:,k]*Vj*Vk,s1[:,j]*s0[:,k]*Vk,s1[:,j]*s0[:,k]*Vj,
                s1[:,j]*s0[:,k]*Vj*Vk,s1[:,j]*s0[:,k]*Vk,s0[:,j]*s1[:,k]*Vk,
                s0[:,j]*s0[:,k],s0[:,j]*s0[:,k]*Vk)
            for name,value in zip(hp.PAIR_KEYS,products):result[name][j,k]=weights@value
    return result


class HermitePairTests(unittest.TestCase):
    def test_normalized_recurrence_and_orthogonality(self):
        z,w=np.polynomial.hermite_e.hermegauss(16);w/=math.sqrt(2*math.pi)
        H=hp.normalized_hermites(z,8)
        np.testing.assert_allclose(H[:,2],(z*z-1)/math.sqrt(2),rtol=2e-15,atol=1e-14)
        np.testing.assert_allclose(H.T@(w[:,None]*H),np.eye(9),rtol=1e-12,atol=1e-12)

    def test_scalar_bank_matches_exact_polynomial_coefficients(self):
        biases=np.array([.2,-.3,0.,1.]);scales=np.array([.6,1.2,0.,.1])
        for degree in (64,128):
            for order in (16,24):
                bank=hp.scalar_bank(biases,scales,degree,order,derivative_fn=polynomial)
                exact=polynomial_bank(biases,scales,degree)['coefficients']
                # All true cubic coefficients agree tightly. The high-degree
                # aliases at N128/q16 are checked explicitly below, not hidden
                # by a loose coefficient tolerance or the energy flags.
                np.testing.assert_allclose(bank['coefficients'][:,:,:4],exact[:,:,:4],rtol=3e-12,atol=5e-13)
                if degree==64 or order==24:
                    np.testing.assert_allclose(bank['coefficients'],exact,rtol=3e-12,atol=5e-11)
                self.assertFalse(bank['material_negative_energy'].any())
                self.assertEqual(bank['node_counts'][2],0)

    def test_order_comparison_detects_high_degree_polynomial_aliases(self):
        biases=np.array([.2,-.3,0.,1.]);scales=np.array([.6,1.2,0.,.1])
        banks=[hp.scalar_bank(biases,scales,128,q,derivative_fn=polynomial) for q in (16,24)]
        errors=[float(np.max(np.abs(bank['coefficients'][:,:,4:]))) for bank in banks]
        self.assertGreater(errors[0],1e-11)
        self.assertLess(errors[0],1e-7)
        self.assertLess(errors[1],5e-11)
        self.assertLess(errors[1],errors[0]/100)
        # Small squared alias energy can escape the negative-energy flags.
        self.assertTrue(all(bank['all_estimates_valid'] for bank in banks))

    def test_all_ten_ordered_fields_match_direct_gaussian_polynomials(self):
        pi,p0,nu,v0=synthetic_case();geometry=hp.covariance_geometry(pi,nu)
        bank=polynomial_bank(p0,geometry['scales'])
        K,tails=hp.gaussian_kernels(bank,geometry['rho'])
        got=hp.affine_pair_fields(K,v0,geometry['B'],geometry['C'])
        expected=direct_polynomial_fields(pi,p0,nu,v0)
        for name in hp.PAIR_KEYS:
            with self.subTest(field=name):np.testing.assert_allclose(got[name],expected[name],rtol=3e-12,atol=2e-12)
        np.testing.assert_array_equal(got['p10Vk'],got['v10Vk'])
        np.testing.assert_allclose(got['G'],got['G'].T,rtol=2e-14,atol=2e-13)
        np.testing.assert_allclose(got['p11VV'],got['p11VV'].T,rtol=2e-14,atol=2e-13)
        np.testing.assert_allclose(got['p10Vj'],got['v01Vk'].T,rtol=2e-14,atol=2e-13)
        self.assertTrue(all(np.all(value==0) for value in tails.values()))

    def test_requested_row_order_and_zero_gate_limits(self):
        pi,p0,nu,v0=synthetic_case();geometry=hp.covariance_geometry(pi,nu)
        bank=polynomial_bank(p0,geometry['scales']);rows=np.array([3,1,0])
        K,_=hp.gaussian_kernels(bank,geometry['rho'],rows)
        got=hp.affine_pair_fields(K,v0,geometry['B'],geometry['C'],rows)
        expected=direct_polynomial_fields(pi,p0,nu,v0)
        for name in hp.PAIR_KEYS:np.testing.assert_allclose(got[name],expected[name][rows],rtol=3e-12,atol=2e-12)
        self.assertTrue(geometry['zero_scale'][3]);np.testing.assert_array_equal(geometry['rho'][3],np.zeros(4))

    def test_correlation_damped_tail_and_absolute_propagation(self):
        pi,p0,nu,v0=synthetic_case();geometry=hp.covariance_geometry(pi,nu)
        bank=polynomial_bank(p0,geometry['scales']);bank['residual_energy_estimate']=np.arange(20).reshape(4,5)/10
        _,tails=hp.gaussian_kernels(bank,geometry['rho'])
        expected=np.abs(geometry['rho'])**65*np.sqrt(bank['residual_energy_estimate'][:,0,None])*np.sqrt(bank['residual_energy_estimate'][None,:,1])
        np.testing.assert_allclose(tails[0,1],expected,rtol=2e-15,atol=0)
        propagated=hp.affine_pair_fields(tails,v0,geometry['B'],geometry['C'],absolute_coefficients=True)
        signs={key:value*np.where(np.indices(value.shape).sum(axis=0)%2,1.,-1.) for key,value in tails.items()}
        signed=hp.affine_pair_fields(signs,v0,geometry['B'],geometry['C'])
        for name in hp.PAIR_KEYS:self.assertTrue(np.all(np.abs(signed[name])<=propagated[name]+1e-12))

    def test_material_negative_energy_is_preserved_and_invalidates_estimate(self):
        def bad_rule(*args,**kwargs):return np.array([0.]),np.array([2.])
        def constant(x,max_order=4):return [np.ones_like(x)]+[np.zeros_like(x)]*4
        bank=hp.scalar_bank([0.],[1.],64,16,derivative_fn=constant,rule_fn=bad_rule)
        self.assertLess(bank['signed_residual_energy'][0,0],0)
        self.assertTrue(bank['material_negative_energy'][0,0])
        self.assertTrue(np.isnan(bank['residual_energy_estimate'][0,0]))
        self.assertFalse(bank['all_estimates_valid'])
        _,tails=hp.gaussian_kernels(bank,np.ones((1,1)))
        self.assertTrue(np.isnan(tails[0,0][0,0]))

    def test_invalid_settings_and_underflow_do_not_create_hidden_floors(self):
        for degree,order in [(32,16),(64,8),(True,16),(64,True)]:
            with self.assertRaises((ValueError,TypeError)):hp.scalar_bank([0.],[1.],degree,order)
        with self.assertRaises(hp.DiagnosticFailure):hp.covariance_geometry(np.array([[1e-300]]),np.array([[1.]]))
        with self.assertRaises(ValueError):hp.scalar_bank([0.],[-1.])
        with self.assertRaises(ValueError):hp.gaussian_kernels(polynomial_bank([0.],[1.]),np.array([[1.01]]))

    def test_fake_base_keeps_diagonal_teacher_and_labels_nonexact_gradient(self):
        class FakeReference:
            def __init__(self):self.quad_limit=12.
            def diag_terms(self,*args):return 'same diagonal'
            def teacher_terms(self,*args):return 'same teacher'
            def evaluate(self,*args,**kwargs):return {'fake':True}
        klass=hp.make_hermite_evaluator_class(FakeReference)
        self.assertIs(klass.diag_terms,FakeReference.diag_terms)
        self.assertIs(klass.teacher_terms,FakeReference.teacher_terms)
        instance=klass();result=instance.evaluate()
        self.assertFalse(result['training_authorized'])
        self.assertIn('NOT exact finite-series-loss',result['gradient_semantics'])
        self.assertIsNone(instance._hermite_context)

    def test_no_population_imports_in_core(self):
        import ast
        tree=ast.parse(Path(hp.__file__).read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module)
        self.assertFalse(set(names)&{'swpop','student_transition','student_transition_fast','teacher_h3'})

if __name__=='__main__':unittest.main()
