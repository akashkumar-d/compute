"""Deterministic Gaussian cross moments for six additional smooth scalar links.

Only the teacher cross kernel is approximated here; the existing student Gram,
AGOP, refit, and update rules are unchanged. Threshold-aware piecewise Gaussian
integration remains resolved even when a student row becomes teacher-parallel.
The spatial gradient uses Stein's identity, avoiding inverse residual variance.
See ADAPTER_REVIEW.md and verify_adapter.py for independent error checks.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import ndtr, expit, erf

SMOOTH_LINKS=('softplus','silu','gelu','tanh','erf','gaussian_rbf')
MOMENTS=json.loads(Path(__file__).with_name('canonical14_moments.json').read_text())['links']
SQRT2PI=np.sqrt(2*np.pi)
BASE_CUTS=np.array([-12.,-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.,12.])
TRANSITION_CUTS=np.array([-12.,-8.,-4.,-2.,-1.,0.,1.,2.,4.,8.,12.])

def density(z): return np.exp(-.5*z*z)/SQRT2PI

def scalar_link(name,z):
    z=np.asarray(z)
    if name=='softplus': return np.logaddexp(0.,z)
    if name=='silu': return z*expit(z)
    if name=='gelu': return z*ndtr(z)
    if name=='tanh': return np.tanh(z)
    if name=='erf': return erf(z)
    if name=='gaussian_rbf': return np.exp(-.5*z*z)
    raise ValueError(name)

def scalar_derivative(name,z):
    z=np.asarray(z)
    if name=='softplus': return expit(z)
    if name=='silu':
        s=expit(z); return s+z*s*(1-s)
    if name=='gelu': return ndtr(z)+z*density(z)
    if name=='tanh': return 1-np.tanh(z)**2
    if name=='erf': return 2/np.sqrt(np.pi)*np.exp(-z*z)
    if name=='gaussian_rbf': return -z*np.exp(-.5*z*z)
    raise ValueError(name)

@lru_cache(None)
def legendre(order): return np.polynomial.legendre.leggauss(order)

@lru_cache(None)
def normal_rule(order):
    # Piecewise rather than Hermite: the same smooth-rule accuracy is retained
    # for saturated sigmoid links and arbitrarily shifted conditional means.
    x,w=legendre(order)
    lo,hi=BASE_CUTS[:-1,None],BASE_CUTS[1:,None]
    z=((hi-lo)/2*x+(hi+lo)/2).ravel()
    weight=((hi-lo)/2*w).ravel()*density(z)
    return z,weight

class SmoothTeacher:
    def __init__(self,name,r,d,order=16,conditional_order=16):
        if name not in SMOOTH_LINKS: raise ValueError(name)
        if not 1<=r<=d: raise ValueError('Require 1 <= r <= d')
        self.name,self.r,self.d=name,r,d
        self.U=np.eye(d)[:,:r];self.lam=np.ones(r)/np.sqrt(r)
        row=MOMENTS[name]
        self.raw_mean=row['raw_mean'];self.raw_second=row['raw_second'];self.raw_zq=row['raw_zq']
        self.norm=np.sqrt(self.raw_second+(r-1)*self.raw_mean**2)
        self.mean=np.sqrt(r)*self.raw_mean/self.norm
        self.second_moment=1.;self.variance=1-self.mean**2
        self.order=int(order);self.conditional_order=int(conditional_order)
        self.numerical_method={'method':'threshold-aware piecewise Gauss-Legendre teacher cross; Stein spatial gradient',
          'integration_interval':[-12.,12.],'base_cuts':BASE_CUTS.tolist(),
          'transition_cuts_in_residual_sd':TRANSITION_CUTS.tolist(),
          'legendre_order_per_segment':self.order,'conditional_order_per_segment':self.conditional_order,
          'scalar_normalization':'fixed canonical14_moments.json; raw mean retained',
          'error_status':'validated numerical quadrature, not certified population error bounds'}

    def raw_values(self,z): return scalar_link(self.name,z)
    def values(self,X): return self.raw_values(np.asarray(X)@self.U)@self.lam/self.norm

    def cross(self,W,b,alpha,grad=True):
        W=np.asarray(W,dtype=float);b=np.asarray(b,dtype=float)
        if W.ndim!=2 or W.shape!=(len(b),self.d):raise ValueError('Wrong W,b shape')
        n=np.linalg.norm(W,axis=1)
        if np.min(n)<1e-150:raise FloatingPointError('Near-zero spatial row')
        C=np.zeros(len(b));gb=np.zeros(len(b));gw=np.zeros_like(W)
        fac=1/(np.sqrt(self.r)*self.norm)
        if alpha==1:
            C=fac*(self.r*b*self.raw_mean+self.raw_zq*W[:,:self.r].sum(axis=1))
            if grad:gb[:]=fac*self.r*self.raw_mean;gw[:,:self.r]=fac*self.raw_zq
            return C,gw,gb
        J=1-alpha; beta=b/n
        x,weight=legendre(self.order)
        nz,nw=normal_rule(self.conditional_order)
        for i in range(self.r):
            # Direct residual-coordinate norm avoids cancellation in 1-rho^2.
            residual=W.copy();residual[:,i]=0.
            v=np.linalg.norm(residual,axis=1)/n
            rho=W[:,i]/n
            critical=np.divide(-beta,rho,out=np.zeros_like(beta),where=rho!=0)
            width=np.divide(v,np.abs(rho),out=np.full_like(v,1e30),where=rho!=0)
            moving=np.clip(critical[:,None]+width[:,None]*TRANSITION_CUTS,-12.,12.)
            cuts=np.sort(np.concatenate((np.broadcast_to(BASE_CUTS,(len(b),len(BASE_CUTS))),moving),axis=1),axis=1)
            lo,hi=cuts[:,:-1,None],cuts[:,1:,None]
            z=(hi-lo)/2*x+(hi+lo)/2
            qw=(hi-lo)/2*weight*density(z)
            mu=beta[:,None,None]+rho[:,None,None]*z
            arg=np.divide(mu,v[:,None,None],out=np.zeros_like(mu),where=v[:,None,None]>0)
            Phi=np.where(v[:,None,None]>0,ndtr(arg),(mu>0).astype(float))
            positive=np.where(v[:,None,None]>0,v[:,None,None]*density(arg)+mu*Phi,np.maximum(mu,0.))
            H=n[:,None,None]*(alpha*mu+J*positive)
            p=alpha+J*Phi
            q=self.raw_values(z)
            C+=fac*np.sum(qw*q*H,axis=(1,2))
            if grad:
                gb+=fac*np.sum(qw*q*p,axis=(1,2))
                # E[X q(Z) sigma'(W.X+b)] = u E[q'(Z) sigma']
                #     + W J phi(beta)/||W|| E[q(Z)| W.X=-b].
                direct=np.sum(qw*scalar_derivative(self.name,z)*p,axis=(1,2))
                conditional=np.sum(self.raw_values(-rho[:,None]*beta[:,None]+v[:,None]*nz)*nw,axis=1)
                delta=J*density(beta)/n*conditional
                gw+=fac*delta[:,None]*W
                gw[:,i]+=fac*direct
        return C,gw,gb
