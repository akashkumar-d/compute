"""Bounded single-thread numerical checks; no training is executed."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import hashlib,importlib.util,json,math,time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,eval_hermitenorm
from population import Teacher,loss_force
from teachers_smooth import SmoothTeacher,SMOOTH_LINKS,MOMENTS,scalar_link,scalar_derivative,density
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[2]/'goal_followup_v6/relu_confirmation/code'
started=time.monotonic();report={};rng=np.random.default_rng(17631)

def scalar(name,z):
    if name in SMOOTH_LINKS:return float(scalar_link(name,z))
    if name.startswith('h'):return float(eval_hermitenorm(int(name[1:]),z)/math.sqrt(math.factorial(int(name[1:]))))
    if name=='relu':return max(z,0.)
    if name=='leaky_relu':return max(z,0.)-.1*max(-z,0.)
    if name=='abs':return abs(z)
    if name=='sine':return np.sin(z)
    raise ValueError(name)

def intall(fn,cuts=[]):
    bounds=sorted(set([-np.inf]+[float(c) for c in cuts if np.isfinite(c)]+[np.inf]))
    return sum(quad(fn,a,b,epsabs=2e-12,epsrel=2e-12,limit=160)[0] for a,b in zip(bounds[:-1],bounds[1:]))

mom=[]
for name,row in MOMENTS.items():
    vals=[intall(lambda z,k=k:scalar(name,z)**k*density(z),[0.]) for k in [1,2]]
    vals.append(intall(lambda z:z*scalar(name,z)*density(z),[0.]))
    errors=[abs(v-row[key]) for v,key in zip(vals,['raw_mean','raw_second','raw_zq'])]
    mom.append({'teacher':name,'errors':errors})
report['scalar_moments']=mom
assert max(max(x['errors']) for x in mom)<2e-12

# Exact prior implementations are bytewise compared at the same states.
spec=importlib.util.spec_from_file_location('oldextra',OLD/'teachers_extra.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('oldpop',OLD/'population.py');op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
W=rng.normal(size=(7,5));b=rng.normal(size=7)
oldchecks=[]
for name in ['h2','h3','h4','sine','relu','abs','h3_plus_h5','h3_plus_sine']:
    oldt=op.MixtureTeacher(name,2,5) if name=='h3' else old.ExtraTeacher(name,2,5)
    newt=Teacher(name,2,5)
    for alpha in [0.,.1,1.]:
        a=oldt.cross(W,b,alpha);bb=newt.cross(W,b,alpha)
        assert all(np.array_equal(x,y) for x,y in zip(a,bb)),name
    oldchecks.append(name)
report['old_exact_bitwise_equal']=oldchecks

# Independent direct conditional integration at zero, small, and extreme angles.
ref=[]
for name in list(SMOOTH_LINKS)+['h5','leaky_relu']:
    t=Teacher(name,1,2)
    for rho,res,beta in [(0.,1.,0.),(.6,.8,.75),(-.6,.8,-.75),(1.,0.,.75),(-1.,0.,-.75),(1.,1e-10,-.3),(-1.,1e-7,2.),(.8,.6,7.),(.8,.6,-9.)]:
        norm=math.hypot(rho,res);rho/=norm;res/=norm
        w=np.array([[rho,res]]);bias=np.array([beta])
        for alpha in [0.,.1,1.]:
            C,G,B=t.cross(w,bias,alpha)
            if res:
                cut=-beta/rho if rho else 0.;width=res/abs(rho) if rho else 1.
                cuts=[0.]+[cut+a*width for a in [-12,-8,-4,-1,0,1,4,8,12]]
            else:cuts=[0.,-beta/rho]
            def p(z):return alpha+(1-alpha)*(ndtr((beta+rho*z)/res) if res else float(beta+rho*z>0))
            def H(z):
                u=beta+rho*z
                return alpha*u+(1-alpha)*(res*density(u/res)+u*ndtr(u/res) if res else max(u,0.))
            R0=intall(lambda z:scalar(name,z)*H(z)*density(z),cuts)/t.norm
            RB=intall(lambda z:scalar(name,z)*p(z)*density(z),cuts)/t.norm
            RG0=intall(lambda z:z*scalar(name,z)*p(z)*density(z),cuts)/t.norm
            # Exact Gaussian conditioning for the residual-coordinate derivative.
            cond=intall(lambda z:scalar(name,-rho*beta+res*z)*density(z),[0.])
            RG1=(1-alpha)*res*density(beta)*cond/t.norm
            errs=[abs(C[0]-R0),abs(B[0]-RB),abs(G[0,0]-RG0),abs(G[0,1]-RG1)]
            ref.append({'teacher':name,'rho':rho,'residual':res,'beta':beta,'alpha':alpha,'max_abs_error':max(errs)})
report['independent_cross']=ref
assert max(x['max_abs_error'] for x in ref)<2e-10,max(ref,key=lambda x:x['max_abs_error'])

# Whole loss and cross finite differences, including profiled-mean correction.
fd=[]
for name in MOMENTS:
    t=Teacher(name,2,5);W=rng.normal(size=(3,5));b=rng.normal(size=3);A=rng.normal(size=3)
    C,G,B=t.cross(W,b,.1);err=0.
    eps=2e-6
    for j,k in [(0,0),(0,4),(2,1)]:
        wp=W.copy();wm=W.copy();wp[j,k]+=eps;wm[j,k]-=eps
        obs=(t.cross(wp,b,.1,False)[0][j]-t.cross(wm,b,.1,False)[0][j])/(2*eps)
        err=max(err,abs(obs-G[j,k]))
    bp=b.copy();bm=b.copy();bp[1]+=eps;bm[1]-=eps
    obs=(t.cross(W,bp,.1,False)[0][1]-t.cross(W,bm,.1,False)[0][1])/(2*eps);err=max(err,abs(obs-B[1]))
    for profiled in [False,True]:
        L,F,*_=loss_force(A,W,b,t,.1,profiled)
        wp=W.copy();wm=W.copy();wp[0,4]+=eps;wm[0,4]-=eps
        obs=(loss_force(A,wp,b,t,.1,profiled)[0]-loss_force(A,wm,b,t,.1,profiled)[0])/(2*eps)
        err=max(err,abs(obs+2/len(A)*F[1][0,4]))
    fd.append({'teacher':name,'max_abs_error':err})
report['finite_differences']=fd
assert max(x['max_abs_error'] for x in fd)<2e-8

# Order and scale stress. Independent reference is scale-equivariant.
conv=[]
for name in SMOOTH_LINKS:
    W=rng.normal(size=(5,5));b=np.array([0.,.1,1.,-3.,7.])*np.linalg.norm(W,axis=1)
    W[1]=[1.,1e-12,0,0,0];b[1]=-.3
    vals=[]
    for order in [8,16,32]:vals.append(SmoothTeacher(name,2,5,order,order).cross(W,b,.1))
    e16=max(np.max(np.abs(a-bb)) for a,bb in zip(vals[1],vals[2]))
    e8=max(np.max(np.abs(a-bb)) for a,bb in zip(vals[0],vals[2]))
    t=SmoothTeacher(name,2,5);C,G,B=t.cross(W,b,.1)
    scaleerr=0.
    for scale in [1e-9,1e-4,1e4]:
        c,g,bb=t.cross(W*scale,b*scale,.1)
        scaleerr=max(scaleerr,np.max(np.abs(c/scale-C)),np.max(np.abs(g-G)),np.max(np.abs(bb-B)))
    conv.append({'teacher':name,'order8_vs32':float(e8),'order16_vs32':float(e16),'scale_covariance_error':float(scaleerr)})
report['convergence']=conv
assert max(x['order16_vs32'] for x in conv)<2e-10
assert max(x['scale_covariance_error'] for x in conv)<2e-12

# Typical batch evaluation timing, not a trajectory/training run.
W=rng.normal(size=(128,64))*.01/8;b=rng.normal(size=128)*.01/8
bench=[]
for name in SMOOTH_LINKS:
    t=Teacher(name,8,64);tic=time.monotonic();t.cross(W,b,0.);elapsed=time.monotonic()-tic
    bench.append({'teacher':name,'m':128,'d':64,'r':8,'one_cross_seconds':elapsed})
report['batch_timing']=bench
report['status']='PASS';report['elapsed_seconds']=time.monotonic()-started
report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
(ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k in ['status','elapsed_seconds','batch_timing','convergence']},indent=2))
