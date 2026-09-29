"""Audit-only N256/q32 clone of reviewed Hermite moments; never the training evaluator.

Requires the pinned scalar transition_rule on the caller's import path.
Finite-series moments approximate the intended Gaussian expectations. Inherited
population-gradient formulas are NOT exact derivatives of the finite-series
loss. No training/adaptive-mask interface or accuracy certificate is provided.
"""
from __future__ import annotations
import math
import operator
import numpy as np
from transition_rule import transition_nodes, silu_derivatives

PAIR_KEYS = ('G','pSSVV2','p11VV','p10Vk','p10Vj','p10VV',
             'v10Vk','v01Vk','v00','v00Vk')
KERNEL_KEYS = tuple((r,s) for r in range(5) for s in range(5-r))
CORRELATION_ROUNDOFF = 64*np.finfo(float).eps
NEGATIVE_ENERGY_RTOL = 1e-10


class DiagnosticFailure(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _integer(value, allowed, name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+' must be an integer')
    value=operator.index(value)
    if value not in allowed:raise ValueError(name+' must be one of '+str(tuple(allowed)))
    return value


def normalized_hermites(z, degree):
    """h_n=He_n/sqrt(n!), using the normalized three-term recurrence."""
    z=np.asarray(z,dtype=float)
    if z.ndim!=1 or not np.isfinite(z).all() or degree<0:
        raise ValueError('Finite one-dimensional nodes and nonnegative degree required')
    H=np.empty((len(z),degree+1));H[:,0]=1.
    if degree:H[:,1]=z
    for n in range(1,degree):
        H[:,n+1]=(z*H[:,n]-math.sqrt(n)*H[:,n-1])/math.sqrt(n+1)
    return H


def scalar_bank(biases, scales, degree=256, order=32, limit=12.,
                derivative_fn=silu_derivatives, rule_fn=transition_nodes):
    """O(m) marginal banks. Raw signed energy residuals are always retained."""
    degree=_integer(degree,(256,),'degree')
    order=_integer(order,(32,),'order')
    biases,scales=np.asarray(biases,float),np.asarray(scales,float)
    if (biases.ndim!=1 or scales.shape!=biases.shape or not np.isfinite(biases).all()
            or not np.isfinite(scales).all() or np.any(scales<0)
            or not math.isfinite(limit) or limit<=0):
        raise ValueError('Finite matched bias/nonnegative-scale vectors and positive limit required')
    m=len(biases);coefficients=np.zeros((m,5,degree+1));energy=np.empty((m,5))
    node_counts=np.zeros(m,dtype=int)
    for j,(bias,scale) in enumerate(zip(biases,scales)):
        if scale==0:
            value=np.asarray(derivative_fn(np.array([bias]),max_order=4),float)[:,0]
            coefficients[j,:,0]=value;energy[j]=value*value
            continue
        z,w=rule_fn(bias,scale,order=order,limit=limit)
        derivatives=np.asarray(derivative_fn(bias+scale*z,max_order=4),float)
        if derivatives.shape!=(5,len(z)) or not np.isfinite(derivatives).all():
            raise ValueError('Derivative function must return five finite derivative arrays')
        H=normalized_hermites(z,degree)
        coefficients[j]=(derivatives*w)@H
        energy[j]=(derivatives*derivatives)@w
        node_counts[j]=len(z)
    if not np.isfinite(coefficients).all() or not np.isfinite(energy).all():
        raise DiagnosticFailure('Nonfinite scalar bank',dict(stage='scalar_bank'))
    retained=np.sum(coefficients*coefficients,axis=2)
    residual=energy-retained
    tolerance=NEGATIVE_ENERGY_RTOL*np.abs(energy)+256*np.finfo(float).eps*retained
    negative=residual<0
    material_negative=residual < -tolerance
    # This is an explicitly labeled estimate, never a rigorous residual bound.
    # Only immaterial negative roundoff is rounded to zero; material failures
    # receive NaN estimates and retain their signed raw energy values/flags.
    residual_estimate=np.maximum(residual,0.)+tolerance
    residual_estimate[material_negative]=np.nan
    return dict(coefficients=coefficients,energy=energy,retained_energy=retained,
        signed_residual_energy=residual,negative_energy=negative,
        material_negative_energy=material_negative,negative_energy_tolerance=tolerance,
        residual_energy_estimate=residual_estimate,node_counts=node_counts,
        degree=degree,order=order,limit=float(limit),exact_zero_scale=scales==0,
        all_estimates_valid=bool(not material_negative.any()),
        estimate_note='Finite quadrature residual estimate; not a certified Hermite or Gaussian-tail error bound')


def covariance_geometry(pi,nu):
    """Actual marginal Gaussian covariances; no conditional residual floor/drop."""
    pi,nu=np.asarray(pi,float),np.asarray(nu,float)
    if pi.ndim!=2 or nu.shape!=pi.shape or not np.isfinite(pi).all() or not np.isfinite(nu).all():
        raise ValueError('Matching finite (dimension,width) parameter arrays required')
    A=pi.T@pi;B=nu.T@pi;C=nu.T@nu
    if not all(np.isfinite(x).all() for x in (A,B,C)):
        raise DiagnosticFailure('Nonfinite Gaussian covariance',dict(stage='covariance'))
    norm2=np.diag(A)
    zero=np.all(pi==0.,axis=0)
    if np.any((norm2<=0)&~zero):
        raise DiagnosticFailure('Unresolved nonzero gate scale; no hidden floor',
                                dict(stage='gate_scale',unresolved_indices=np.flatnonzero((norm2<=0)&~zero)))
    scales=np.sqrt(norm2);denominator=np.outer(scales,scales)
    nonzero=~(zero[:,None]|zero[None,:])
    if np.any(nonzero & ((denominator<=0)|~np.isfinite(denominator))):
        raise DiagnosticFailure('Unresolved correlation denominator',dict(stage='correlation_denominator'))
    rho=np.zeros_like(A);rho[nonzero]=A[nonzero]/denominator[nonzero]
    overshoot=np.maximum(np.abs(rho)-1.,0.)
    if np.any(overshoot>CORRELATION_ROUNDOFF):
        raise DiagnosticFailure('Correlation outside declared roundoff allowance',dict(
            stage='correlation',max_overshoot=float(overshoot.max()),allowance=CORRELATION_ROUNDOFF))
    clipped=overshoot>0
    rho=np.clip(rho,-1.,1.)
    return dict(A=A,B=B,C=C,scales=scales,rho=rho,zero_scale=zero,
        correlation_clipped=clipped,max_correlation_overshoot=float(overshoot.max(initial=0.)),
        correlation_roundoff_allowance=CORRELATION_ROUNDOFF,
        geometry_note='Actual input marginal covariances; removes reference basis reorthogonalization rounding and tiny residual-drop approximation. No new physical floor.')


def gaussian_kernels(bank,rho,rows=None):
    """K_rs via Hermite contraction, plus correlation-damped tail estimates."""
    coefficients=np.asarray(bank['coefficients']);m=len(coefficients)
    rho=np.asarray(rho,float)
    if rho.shape!=(m,m) or not np.isfinite(rho).all() or np.any(np.abs(rho)>1):
        raise ValueError('Finite valid correlation matrix required')
    rows=np.arange(m) if rows is None else np.asarray(rows,dtype=int)
    if rows.ndim!=1 or np.any(rows<0) or np.any(rows>=m):raise ValueError('Invalid row indices')
    degree=coefficients.shape[2]-1
    powers=rho[rows,:,None]**np.arange(degree+1)[None,None,:]
    damp=np.abs(rho[rows])**(degree+1)
    residual=np.sqrt(bank['residual_energy_estimate'])
    kernels={};tails={}
    for r,s in KERNEL_KEYS:
        kernels[r,s]=np.einsum('jn,kn,jkn->jk',coefficients[rows,r],coefficients[:,s],powers,optimize=True)
        tails[r,s]=damp*residual[rows,r,None]*residual[None,:,s]
    return kernels,tails


def affine_pair_fields(kernels,v0,B,C,rows=None,absolute_coefficients=False):
    """Exact Gaussian affine-value identities with the original key orientation.

    When absolute_coefficients=True and kernels contains nonnegative K error
    estimates, returns their triangle-inequality propagation, not signed fields.
    """
    v0=np.asarray(v0,float);B=np.asarray(B,float);C=np.asarray(C,float);m=len(v0)
    if B.shape!=(m,m) or C.shape!=(m,m):raise ValueError('Covariance shape mismatch')
    rows=np.arange(m) if rows is None else np.asarray(rows,dtype=int)
    dj,dk=v0[rows,None],v0[None,:]
    bjj=np.diag(B)[rows,None];bjk=B[rows,:]
    bkj=B[:,rows].T;bkk=np.diag(B)[None,:]
    def coefficient(value):return np.abs(value) if absolute_coefficients else value
    def one(r,s,which):
        d,x,y=(dj,bjj,bjk) if which=='j' else (dk,bkj,bkk)
        return coefficient(d)*kernels[r,s]+coefficient(x)*kernels[r+1,s]+coefficient(y)*kernels[r,s+1]
    def two(r,s):
        return (coefficient(dj*dk+C[rows,:])*kernels[r,s]
            +coefficient(dj*bkj+dk*bjj)*kernels[r+1,s]
            +coefficient(dj*bkk+dk*bjk)*kernels[r,s+1]
            +coefficient(bjj*bkj)*kernels[r+2,s]
            +coefficient(bjj*bkk+bjk*bkj)*kernels[r+1,s+1]
            +coefficient(bjk*bkk)*kernels[r,s+2])
    shared=one(1,0,'k')
    return dict(G=two(0,0),pSSVV2=two(2,0),p11VV=two(1,1),
        p10Vk=shared,p10Vj=one(1,0,'j'),p10VV=two(1,0),
        v10Vk=shared.copy(),v01Vk=one(0,1,'k'),v00=kernels[0,0].copy(),v00Vk=one(0,0,'k'))


def make_hermite_evaluator_class(reference_class):
    """Opt-in diagnostic subclass, preserving inherited diagonal/teacher methods.

    Caller imports the reviewed reference model explicitly on the server.
    This module and its synthetic tests never import a population model.
    """
    class HermiteDiagnosticSwiGLUPop(reference_class):
        gradient_semantics='Inherited population-gradient formulas approximate intended gradient; NOT exact finite-series-loss derivatives'
        training_authorized=False

        def __init__(self,*args,hermite_degree=256,hermite_quad_order=32,**kwargs):
            self.hermite_degree=_integer(hermite_degree,(256,),'hermite_degree')
            self.hermite_quad_order=_integer(hermite_quad_order,(32,),'hermite_quad_order')
            super().__init__(*args,**kwargs)
            self._hermite_context=None;self._hermite_in_evaluate=False
            self.hermite_diagnostics=None

        def _context(self,pi,p0,nu,v0):
            token=tuple(id(x) for x in (pi,p0,nu,v0))
            if self._hermite_in_evaluate and self._hermite_context is not None:
                if self._hermite_context['token']!=token:raise RuntimeError('Unexpected parameter views within one evaluation')
                return self._hermite_context
            geometry=covariance_geometry(pi,nu)
            bank=scalar_bank(p0,geometry['scales'],self.hermite_degree,self.hermite_quad_order,self.quad_limit)
            m=len(p0)
            diagnostics=dict(degree=self.hermite_degree,scalar_order=self.hermite_quad_order,
                scalar_limit=self.quad_limit,marginal_banks=m,gradient_semantics=self.gradient_semantics,
                training_authorized=False,adaptive_masks=False,geometry=geometry,
                scalar_bank={k:v for k,v in bank.items() if k!='coefficients'},
                per_pair_tail_estimates={k:np.full((m,m),np.nan) for k in PAIR_KEYS},
                pair_rows_returned=[],all_scalar_estimates_valid=bank['all_estimates_valid'],
                status='diagnostic_only' if bank['all_estimates_valid'] else 'invalid_scalar_energy_estimate')
            context=dict(token=token,geometry=geometry,bank=bank,diagnostics=diagnostics)
            self.hermite_diagnostics=diagnostics
            if self._hermite_in_evaluate:self._hermite_context=context
            return context

        def pair_terms(self,pi,p0,nu,v0,rows=None):
            context=self._context(pi,p0,nu,v0);geometry=context['geometry']
            m=pi.shape[1];rows=np.arange(m) if rows is None else np.asarray(rows,dtype=int)
            kernels,tails=gaussian_kernels(context['bank'],geometry['rho'],rows)
            fields=affine_pair_fields(kernels,v0,geometry['B'],geometry['C'],rows)
            estimates=affine_pair_fields(tails,v0,geometry['B'],geometry['C'],rows,True)
            for name,value in estimates.items():context['diagnostics']['per_pair_tail_estimates'][name][rows]=value
            context['diagnostics']['pair_rows_returned'].extend(rows.tolist())
            return fields

        def evaluate(self,*args,**kwargs):
            if self._hermite_in_evaluate:raise RuntimeError('Nested evaluation is unsupported')
            self._hermite_in_evaluate=True;self._hermite_context=None;self.hermite_diagnostics=None
            try:
                result=super().evaluate(*args,**kwargs)
                result['hermite_diagnostics']=self.hermite_diagnostics
                result['gradient_semantics']=self.gradient_semantics
                result['training_authorized']=False
                return result
            finally:
                self._hermite_in_evaluate=False;self._hermite_context=None
    return HermiteDiagnosticSwiGLUPop
