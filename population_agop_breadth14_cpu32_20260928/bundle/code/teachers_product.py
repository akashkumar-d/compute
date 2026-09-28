"""Deterministic Gaussian population moments for genuine two-coordinate teachers.

Product links have the form q(s)t for independent standard-normal s,t.
The full target is divided by its L2 norm, so E[y**2]=1. The historical
silu_additive is uncentered; silu_centered_additive explicitly subtracts
2 E[SiLU(S)] and uses sqrt(2 Var(SiLU(S))) as its normalization.

ReLU and polynomial products use closed Gaussian formulas. Smooth gates
use deterministic 1D quadrature (not Monte Carlo). A hard-threshold split
handles near alignment. GELU is the exact x Phi(x), not the tanh approximation.
"""
import numpy as np
from scipy.special import expit, ndtr, roots_hermitenorm
from numpy.polynomial.legendre import leggauss

SQRT2PI = np.sqrt(2*np.pi)
PRODUCT_TEACHERS = {
    'silu_product': {'label': r'$\operatorname{SiLU}(s)t$'},
    'relu_product': {'label': r'$\operatorname{ReLU}(s)t$'},
    'silu_additive': {'label': r'$\operatorname{SiLU}(s)+\operatorname{SiLU}(t)$'},
    'gelu_product': {'label': r'$\operatorname{GELU}(s)t$'},
    'tanh_product': {'label': r'$\tanh(s)t$'},
    'quadratic_product': {'label': r'$h_2(s)t$'},
    'bilinear_product': {'label': r'$st$'},
    'silu_centered_additive': {'label': r'$\operatorname{SiLU}(s)+\operatorname{SiLU}(t)-2\mu$'},
}


def _pdf(x):
    # Beyond |x|=40 the value already underflows to zero in float64. Clipping
    # avoids an irrelevant x**2 overflow in exact-collinearity limits.
    x=np.clip(np.asarray(x),-40.,40.)
    return np.exp(-x*x/2)/SQRT2PI


def _silu(x):
    return x*expit(x)


def _silu_prime(x):
    p=expit(x)
    return p+x*p*(1-p)


def _gelu(x):
    return x*ndtr(x)


def _gelu_prime(x):
    return ndtr(x)+x*_pdf(x)


def _tanh_prime(x):
    return 1-np.tanh(x)**2


def _h2(x):
    return (x*x-1)/np.sqrt(2.)


class ProductTeacher:
    """Normalized rank-two teacher; ``order`` controls quadrature refinement."""
    def __init__(self,name,r,d,order=128):
        if name not in PRODUCT_TEACHERS: raise ValueError(name)
        if r != 2 or d < 2: raise ValueError('These bivariate teachers require r=2 and d>=2.')
        self.name,self.r,self.d=name,r,d
        self.U=np.eye(d)[:,:2]
        self.lam=np.ones(2)/np.sqrt(2)
        self.order=order
        self.is_additive=name in ('silu_additive','silu_centered_additive')
        self.centered=name=='silu_centered_additive'
        self.q,self.qprime={
            'relu_product': (lambda x: np.maximum(x,0),lambda x: (x>0).astype(float)),
            'gelu_product': (_gelu,_gelu_prime),
            'tanh_product': (np.tanh,_tanh_prime),
            'quadratic_product': (_h2,lambda x: np.sqrt(2.)*x),
            'bilinear_product': (lambda x:x,lambda x:np.ones_like(x)),
        }.get(name,(_silu,_silu_prime))
        self._gh,self._ghw=roots_hermitenorm(order)
        self._ghw=self._ghw/SQRT2PI
        self._gx,self._gw=leggauss(max(48,order//2))
        # Constant norms use a higher fixed order independent of runtime order.
        nx,nw=roots_hermitenorm(256);nw=nw/SQRT2PI
        if name == 'relu_product':
            self.link_mean=1/SQRT2PI;self.link_second=.5
        elif name == 'gelu_product':
            self.link_mean=1/(2*np.sqrt(np.pi))
            self.link_second=1/3+1/(2*np.pi*np.sqrt(3))
        elif name in ('quadratic_product','bilinear_product'):
            self.link_mean=0.;self.link_second=1.
        elif name == 'tanh_product':
            self.link_mean=0.;self.link_second=float(nw@(np.tanh(nx)**2))
        else:
            self.link_mean=float(nw@_silu(nx));self.link_second=float(nw@(_silu(nx)**2))
        self.link_prime_mean=float(nw@self.qprime(nx))
        if self.centered:
            self.norm=np.sqrt(2*(self.link_second-self.link_mean**2))
            self.mean=0.
        elif name == 'silu_additive':
            self.norm=np.sqrt(2*self.link_second+2*self.link_mean**2)
            self.mean=2*self.link_mean/self.norm
        else:
            self.norm=np.sqrt(self.link_second);self.mean=0.
        self.second_moment=1.
        self.variance=1-self.mean**2

    def values(self,X):
        X=np.asarray(X)
        if self.is_additive:
            offset=2*self.link_mean if self.centered else 0.
            return (_silu(X[:,0])+_silu(X[:,1])-offset)/self.norm
        q=self.q(X[:,0])
        return q*X[:,1]/self.norm

    def _smooth_gate(self,a,v,b,q,mean):
        """E[q(S) Phi((a*S+b)/v)], stable when v/|a| tends to zero."""
        n=np.hypot(a,v);rho=a/n;out=np.empty_like(a)
        regular=np.abs(rho)<.85
        if np.any(regular):
            ar,vr,br=a[regular],v[regular],b[regular]
            out[regular]=(ndtr((ar[:,None]*self._gh+br[:,None])/vr[:,None])
                          *(q(self._gh)*self._ghw)).sum(axis=1)
        # Split exactly at the limiting halfspace boundary. The correction
        # Phi(u)-1{u>0} lives in [-10,10] and remains smooth on each half.
        for j in np.flatnonzero(~regular):
            s0=-b[j]/a[j];sign=np.sign(a[j]);width=v[j]/abs(a[j])
            cutoff=np.clip(s0,-12.,12.)
            left,right=(cutoff,12.) if sign>0 else (-12.,cutoff)
            z=(left+right)/2+(right-left)/2*self._gx
            value=(right-left)/2*np.dot(self._gw,q(z)*_pdf(z))
            if width>0:
                lo,hi=sorted(((-12-s0)/(sign*width),(12-s0)/(sign*width)))
                lo=max(lo,-10.);hi=min(hi,10.)
                for low,high in ((lo,min(hi,0.)),(max(lo,0.),hi)):
                    if high<=low:continue
                    u=(low+high)/2+(high-low)/2*self._gx
                    s=s0+sign*width*u
                    correction=ndtr(u)-(u>0)
                    value+=width*(high-low)/2*np.dot(self._gw,q(s)*_pdf(s)*correction)
            out[j]=value
        return out

    def _gate(self,W,b,axis,with_prime=False):
        """T=E[q(S)1{Z+b>0}], Tb, gradient T, and E[q'(S)1{Z+b>0}]."""
        n=np.linalg.norm(W,axis=1);a=W[:,axis]
        # Avoid cancellation in n^2-a^2 near a teacher direction.
        v=np.linalg.norm(np.delete(W,axis,axis=1),axis=1)
        beta=b/n;rho=a/n;root=v/n
        cmu=-rho*beta
        if self.name=='relu_product':
            safe=np.maximum(root,1e-300)
            z=cmu/safe
            conditional=root*_pdf(z)+cmu*ndtr(z)
            conditional_prime=ndtr(z)
            T=_pdf(0.)*ndtr(beta/safe)+rho*_pdf(beta)*ndtr(z)
            Tprime=None
        elif self.name=='bilinear_product':
            conditional=cmu
            conditional_prime=np.ones_like(cmu)
            T=rho*_pdf(beta)
            Tprime=ndtr(beta) if with_prime else None
        elif self.name=='quadratic_product':
            # Conditional He_2 moment written without root^2-1 cancellation.
            conditional=rho*rho*(beta*beta-1)/np.sqrt(2.)
            conditional_prime=np.sqrt(2.)*cmu
            T=-rho*rho*beta*_pdf(beta)/np.sqrt(2.)
            Tprime=np.sqrt(2.)*rho*_pdf(beta) if with_prime else None
        else:
            conditional_z=cmu[:,None]+root[:,None]*self._gh
            conditional=self.q(conditional_z)@self._ghw
            conditional_prime=self.qprime(conditional_z)@self._ghw
            T=self._smooth_gate(a,v,b,self.q,self.link_mean)
            Tprime=self._smooth_gate(a,v,b,self.qprime,self.link_prime_mean) if with_prime else None
        density=_pdf(beta)/n
        Tb=density*conditional
        u=np.zeros(self.d);u[axis]=1.
        Tw=density[:,None]*((conditional_prime[:,None]*u)
             - ((b*conditional+a*conditional_prime)/(n*n))[:,None]*W)
        return T,Tb,Tw,Tprime

    def cross(self,W,b,alpha,grad=True):
        W,b=np.asarray(W),np.asarray(b)
        if np.min(np.linalg.norm(W,axis=1))<1e-150:raise FloatingPointError('Near-zero spatial row')
        J=1-alpha
        C=np.zeros(len(b));gw=np.zeros_like(W);gb=np.zeros_like(b)
        if not self.is_additive:
            T,Tb,Tw,_=self._gate(W,b,0)
            gate=alpha*self.link_mean+J*T
            c=W[:,1]
            C=c*gate/self.norm
            if grad:
                gw=c[:,None]*J*Tw/self.norm
                gw[:,1]+=gate/self.norm
                gb=c*J*Tb/self.norm
        else:
            n2=np.sum(W*W,axis=1)
            for axis in (0,1):
                T,Tb,_,Tp=self._gate(W,b,axis,with_prime=True)
                a=W[:,axis]
                # Gaussian Stein: E[q(S) Z 1{Z+b>0}]=n^2 Tb+a Tp.
                C+=(alpha*(.5*a+b*self.link_mean)+J*(b*T+n2*Tb+a*Tp))/self.norm
                if grad:
                    gw+=J*Tb[:,None]*W/self.norm
                    gw[:,axis]+=(alpha*.5+J*Tp)/self.norm
                    gb+=(alpha*self.link_mean+J*T)/self.norm
            if self.centered:
                n=np.sqrt(n2);beta=b/n
                density=_pdf(beta);prob=alpha+J*ndtr(beta)
                offset=2*self.link_mean/self.norm
                C-=offset*(J*n*density+b*prob)
                if grad:
                    gw-=offset*J*(density/n)[:,None]*W
                    gb-=offset*prob
        return C,gw,gb
