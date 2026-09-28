"""Population moments for biased ReLU/leaky networks and full mixture teachers.

No Monte Carlo training: bivariate Gaussian CDFs use scipy's Owen T function.
Network f=sum_j A_j sigma(W_j x+b_j)/m; loss is full MSE.
"""
import numpy as np
from scipy.special import ndtr, owens_t
from scipy.optimize import minimize


def pdf(x):
    return np.exp(-0.5 * np.asarray(x)**2) / np.sqrt(2*np.pi)


def bvn_cdf(a, b, rho, dm=None, dp=None):
    a,b,rho=np.broadcast_arrays(a,b,rho)
    if dm is None: dm=np.maximum(1-rho,0.)
    if dp is None: dp=np.maximum(1+rho,0.)
    true_root=np.sqrt(dm*dp)
    root=np.maximum(true_root,1e-150)
    aa=np.where(np.abs(a)<1e-14,1e-14,a)
    bb=np.where(np.abs(b)<1e-14,1e-14,b)
    offset_ab=np.where(rho>=0,(b-a)+dm*a,(b+a)-dp*a)
    offset_ba=np.where(rho>=0,(a-b)+dm*b,(a+b)-dp*b)
    out=(ndtr(a)+ndtr(b))/2-owens_t(a,offset_ab/(aa*root))-owens_t(b,offset_ba/(bb*root))-(1-np.sign(aa)*np.sign(bb))/4
    za=np.abs(a)<1e-14; zb=np.abs(b)<1e-14
    out=np.where(za,ndtr(b)/2+owens_t(b,rho/root),out)
    out=np.where(zb,ndtr(a)/2+owens_t(a,rho/root),out)
    # atan2 preserves angles smaller than those representable by 1-rho.
    out=np.where(za&zb,np.arctan2(true_root,-rho)/(2*np.pi),out)
    out=np.where((true_root==0)&(rho>=0),ndtr(np.minimum(a,b)),out)
    out=np.where((true_root==0)&(rho<0),np.maximum(ndtr(a)+ndtr(b)-1,0),out)
    return np.clip(out,0,1)


def moments(W,b,alpha):
    m=len(b); J=1-alpha
    norms=np.linalg.norm(W,axis=1)
    if np.min(norms)<1e-150: raise FloatingPointError('Near zero spatial row')
    cov=W@W.T
    # A single symmetric denominator is essential near collinearity: two
    # sequential divisions can differ by an ulp across the diagonal, which
    # becomes a large error after division by sqrt(1-rho**2).
    rho=cov/np.outer(norms,norms)
    rho=np.clip((rho+rho.T)/2,-1,1)
    np.fill_diagonal(rho,1.)
    beta=b/norms; pi=ndtr(beta); fi=pdf(beta)
    unit=W/norms[:,None]
    diff=unit[:,None,:]-unit[None,:,:]
    plus=unit[:,None,:]+unit[None,:,:]
    dm_small=np.sum(diff*diff,axis=2)/2
    dp_small=np.sum(plus*plus,axis=2)/2
    dm=np.where(rho>=0,dm_small,2-dp_small)
    dp=np.where(rho>=0,2-dm_small,dp_small)
    dm=np.clip(dm,0,2);dp=np.clip(dp,0,2)
    true_root=np.sqrt(dm*dp)
    root=np.maximum(true_root,1e-150)
    # Preserve complementary conditional probabilities as rho approaches +/-1.
    conditional_offset=np.where(rho>=0,
        (beta[None,:]-beta[:,None])+dm*beta[:,None],
        (beta[None,:]+beta[:,None])-dp*beta[:,None])
    z=conditional_offset/root
    # A diagonal conditional variance is zero. The convention below yields
    # the continuous same-row limit in the first truncated moments.
    np.fill_diagonal(z,0.)
    c=ndtr(z)
    P=bvn_cdf(beta[:,None],beta[None,:],rho,dm,dp)
    np.fill_diagonal(P,pi)
    P=(P+P.T)/2
    Kpp=(cov+np.outer(b,b))*P + b[:,None]*norms[None,:]*fi[None,:]*c.T + b[None,:]*norms[:,None]*fi[:,None]*c
    Kpp+=np.outer(norms,norms)*true_root*fi[:,None]*pdf(z)
    np.fill_diagonal(Kpp,(norms**2+b**2)*pi+b*norms*fi)
    Kpp=(Kpp+Kpp.T)/2
    plusmean=norms*fi+b*pi
    linearplus=b[:,None]*plusmean[None,:]+cov*pi[None,:]
    K=alpha**2*(cov+np.outer(b,b))+alpha*J*(linearplus+linearplus.T)+J**2*Kpp
    mean=alpha*b+J*plusmean
    Q=alpha**2+alpha*J*(pi[:,None]+pi[None,:])+J**2*P
    Tpp=b[None,:]*P+norms[None,:]*(rho*fi[:,None]*c+fi[None,:]*c.T)
    np.fill_diagonal(Tpp,plusmean)
    Hlinear=b[None,:]*pi[:,None]+cov*(fi/norms)[:,None]
    Db=alpha*mean[None,:]+J*(alpha*Hlinear+J*Tpp)
    muc=norms[None,:]*conditional_offset
    sdc=norms[None,:]*true_root
    posc=sdc*pdf(z)+muc*c
    delta=(fi/norms)[:,None]*(alpha*muc+J*posc)
    np.fill_diagonal(delta,0.)
    return K,Q,Db,delta


TEACHERS={
    'damped_1_2': {'v':[.5], 'p':[1.], 'label':r'$g_{1/2}$'},
    'three_atom': {'v':[.4,.7,1.], 'p':[.2,.3,.5], 'label':'Positive three-scale mixture'},
    'h3': {'v':[1.], 'p':[1.], 'label':r'$h_3$'},
    'damped_2_3': {'v':[2/3], 'p':[1.], 'label':r'$g_{2/3}/\|g_{2/3}\|_2$'},
    'damped_1_3': {'v':[1/3], 'p':[1.], 'label':r'$g_{1/3}/\|g_{1/3}\|_2$'},
    'mixture': {'v':[1.,1/3], 'p':[.5,.5], 'label':r'$(g_1+g_{1/3})/\|g_1+g_{1/3}\|_2$'},
}


class MixtureTeacher:
    def __init__(self,name,r,d):
        self.name=name; self.r=r; self.d=d
        self.v=np.array(TEACHERS[name]['v']);self.p=np.array(TEACHERS[name]['p'])
        t=np.outer(1-self.v,1-self.v)
        kernel=3*(2+3*t)/(1-t)**3.5
        self.norm=np.sqrt(self.p@kernel@self.p)
        self.lam=np.ones(r)/np.sqrt(r)
        self.U=np.eye(d)[:,:r]
        self.mean=0.; self.second_moment=1.

    def cross(self,W,b,alpha,grad=True):
        J=1-alpha; c=W@self.U
        r2=np.sum(W*W,axis=1)
        C=np.zeros(len(b));gb=np.zeros(len(b));gw=np.zeros_like(W)
        for i in range(self.r):
            ci=c[:,i]
            for v,p in zip(self.v,self.p):
                D=r2-(1-v)*ci**2
                F=D**(-1.5)*pdf(b/np.sqrt(D))
                fact=J*p*self.lam[i]/self.norm
                C-=fact*b*ci**3*F
                if grad:
                    gb-=fact*ci**3*F*(1-b*b/D)
                    dw=W-(1-v)*ci[:,None]*self.U[:,i]
                    gw-=fact*(b*F)[:,None]*(3*ci[:,None]**2*self.U[:,i]+(ci**3*(b*b-3*D)/D**2)[:,None]*dw)
        return C,gw,gb

    def values(self,X):
        z=X@self.U;out=np.zeros_like(z)
        for v,p in zip(self.v,self.p):
            out+=p*v**(-3.5)*(z**3-3*v*z)*np.exp(-.5*(1/v-1)*z*z)
        return out@self.lam/self.norm


def loss_force(A,W,b,teacher,alpha):
    m=len(A);K,Q,Db,delta=moments(W,b,alpha)
    C,Cw,Cb=teacher.cross(W,b,alpha)
    f2=A@K@A/m**2; yf=A@C/m
    loss=1-2*yf+f2
    self_w=((1-alpha)*(delta@A)/m)[:,None]*W+(Q*A[None,:])@W/m
    fa=C-K@A/m
    fw=A[:,None]*(Cw-self_w)
    fb=A*(Cb-Db@A/m)
    return loss,(fa,fw,fb),K,Q,C


def normalized_refit(K,C,W,b,alpha,r,previous=None):
    rownorm=np.sqrt(np.sum(W*W,axis=1)+b*b)
    H=K/rownorm[:,None]/rownorm[None,:];c=C/rownorm
    H=(H+H.T)/2
    vals,vecs=np.linalg.eigh(H)
    cutoff=max(vals[-1]*1e-11,1e-14)
    inv=np.where(vals>cutoff,1/np.maximum(vals,cutoff),0)
    v=vecs@(inv*(vecs.T@c))
    l2=32/(1-alpha);l1=64*np.sqrt(r)/(1-alpha)
    if np.linalg.norm(v)>l2 or np.abs(v).sum()>l1:
        # The auxiliary z variables express |v| <= z using linear constraints.
        n=len(c)
        start=v*min(1,l2/max(np.linalg.norm(v),1e-30),l1/max(np.abs(v).sum(),1e-30))
        if previous is not None:start=previous
        cons=[{'type':'ineq','fun':lambda q:l2*l2-q[:n]@q[:n],
               'jac':lambda q:np.r_[-2*q[:n],np.zeros(n)]},
              {'type':'ineq','fun':lambda q:l1-np.sum(q[n:]),
               'jac':lambda q:np.r_[np.zeros(n),-np.ones(n)]},
              {'type':'ineq','fun':lambda q:q[n:]-q[:n],
               'jac':lambda q:np.c_[-np.eye(n),np.eye(n)]},
              {'type':'ineq','fun':lambda q:q[n:]+q[:n],
               'jac':lambda q:np.c_[np.eye(n),np.eye(n)]}]
        result=minimize(lambda q:q[:n]@H@q[:n]-2*c@q[:n],np.r_[start,np.abs(start)],
                        jac=lambda q:np.r_[2*(H@q[:n]-c),np.zeros(n)],constraints=cons,
                        method='SLSQP',options={'ftol':1e-11,'maxiter':500})
        v=result.x[:n]
        # Ensure a feasible diagnostic even if optimization failed to converge.
        v*=min(1,l2/max(np.linalg.norm(v),1e-30),l1/max(np.abs(v).sum(),1e-30))
        status=str(result.message);success=bool(result.success)
    else:status='unconstrained optimum feasible';success=True
    risk=1-2*c@v+v@H@v
    gradient=2*(H@v-c)
    # Convex first-order bound valid for the intersection of the two balls.
    # This exposes failure to optimize the initial comparator accurately.
    lower=risk-gradient@v-min(l2*np.linalg.norm(gradient),l1*np.max(np.abs(gradient)))
    return float(risk),v,{'l1':float(np.abs(v).sum()),'l2':float(np.linalg.norm(v)),
                        'success':success,'message':status,'gram_min':float(vals[0]),
                        'convex_lower_bound':float(max(0.,lower)),
                        'objective_gap_bound':float(risk-max(0.,lower))}


def diagnostics(A,W,b,teacher,alpha,step,h,with_refit=True):
    L,_,K,Q,C=loss_force(A,W,b,teacher,alpha);m=len(A)
    # Scale out max|A| and max||W|| before computing eigendirections.
    As=A/max(np.max(np.abs(A)),1e-150);Ws=W/max(np.linalg.norm(W,axis=1).max(),1e-150)
    G=Ws.T@(Q*np.outer(As,As))@Ws/m**2
    vals,V=np.linalg.eigh((G+G.T)/2)
    top=V[:,-teacher.r:]
    cos2=np.linalg.svd(teacher.U.T@top,compute_uv=False)**2
    relative_gap=float((vals[-teacher.r]-vals[-teacher.r-1])/max(vals[-1],1e-300))
    valid=relative_gap>1e-9 and vals[-teacher.r]/max(vals[-1],1e-300)>1e-9
    row={'step':int(step),'time_eta':float(step*m*h/2),'time_h':float(step*h),
         'loss':float(L),'A_min':float(cos2.min()),'A_mean':float(cos2.mean()),
         'A_sub':float(np.sum((W@teacher.U)**2)/np.sum(W*W)),
         'lambda_r_scaled':float(vals[-teacher.r]),'lambda_next_scaled':float(vals[-teacher.r-1]),
         'relative_gap':relative_gap,'agop_valid':bool(valid),
         'row_norm_max':float(np.linalg.norm(W,axis=1).max()),'head_norm':float(np.linalg.norm(A))}
    if with_refit:
        risk,v,info=normalized_refit(K,C,W,b,alpha,teacher.r)
        row.update(refit=risk,refit_info=info)
    return row


def Teacher(name,r,d):
    if name in TEACHERS:return MixtureTeacher(name,r,d)
    from teachers_product import PRODUCT_TEACHERS,ProductTeacher
    if name in PRODUCT_TEACHERS:
        return ProductTeacher(name,r,d)
    from teachers_extra import ExtraTeacher
    return ExtraTeacher(name,r,d)
