"""Canonical scalar Gaussian H1 links and fixed raw moments for breadth14.

These are uncentered raw functions. Fixed normalization moments are independent
of the student and runtime quadrature order. GELU is exactly z*Phi(z); erf is
erf(z), and RBF is exp(-z*z/2). Nonsmooth weak derivatives use a convention at0
that does not affect Gaussian integrals. No model code is imported here.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erf, expit, ndtr, eval_hermitenorm

SQRT2PI=math.sqrt(2*math.pi)
CANONICAL_NAMES=('h2','h3','h4','h5','relu','leaky_relu','abs','softplus','silu','gelu','sine','tanh','erf','gaussian_rbf')
ALIASES={'he2':'h2','he3':'h3','he4':'h4','he5':'h5','sine':'sine','rbf':'gaussian_rbf','leakyrelu':'leaky_relu'}

def _pdf(z):return np.exp(-.5*np.asarray(z)**2)/SQRT2PI

def _sigmoid_derivative(z):
 p=expit(z);return p*(1-p)

def _silu_prime(z):
 z=np.asarray(z);p=expit(z);return p+z*p*(1-p)

def _gelu_prime(z):return ndtr(z)+np.asarray(z)*_pdf(z)

def _leaky(z):return np.where(np.asarray(z)>=0,np.asarray(z),.1*np.asarray(z))

def _leaky_prime(z):return np.where(np.asarray(z)>0,1.,.1)

def _tanh_prime(z):return 1-np.tanh(z)**2

# Exact raw mean, second moment, and E[weak derivative squared] where known.
CLOSED_MOMENTS={
 **{f'h{k}':(0.,1.,float(k)) for k in (2,3,4,5)},
 'relu':(1/SQRT2PI,.5,.5),
 'leaky_relu':(.9/SQRT2PI,.505,.505),
 'abs':(math.sqrt(2/math.pi),1.,1.),
 'gelu':(1/(2*math.sqrt(math.pi)),1/3+1/(2*math.pi*math.sqrt(3)),1/3+2/(3*math.pi*math.sqrt(3))),
 'sine':(0.,-math.expm1(-2)/2,(1+math.exp(-2))/2),
 'erf':(0.,2/math.pi*math.asin(2/3),4/(math.pi*math.sqrt(5))),
 'gaussian_rbf':(1/math.sqrt(2),1/math.sqrt(3),1/(3*math.sqrt(3))),
}
DEFINITIONS={
 **{f'h{k}':f'He_{k}(z)/sqrt({k}!) (probabilists Hermite)' for k in (2,3,4,5)},
 'relu':'max(z,0)', 'leaky_relu':'z for z>=0;0.1*z otherwise', 'abs':'abs(z)',
 'softplus':'log(1+exp(z)), evaluated as logaddexp(0,z)', 'silu':'z*sigmoid(z)',
 'gelu':'z*Phi(z), exact Gaussian CDF', 'sine':'sin(z)', 'tanh':'tanh(z)',
 'erf':'erf(z)', 'gaussian_rbf':'exp(-z^2/2)',
}

def primitive(name):
 name=ALIASES.get(name,name)
 if name.startswith('h') and name[1:].isdigit() and int(name[1:]) in (2,3,4,5):
  k=int(name[1:]);scale=math.sqrt(math.factorial(k))
  return (lambda z:eval_hermitenorm(k,z)/scale), (lambda z:k*eval_hermitenorm(k-1,z)/scale), []
 table={
  'relu':(lambda z:np.maximum(z,0.),lambda z:(np.asarray(z)>0).astype(float),[0.]),
  'leaky_relu':(_leaky,_leaky_prime,[0.]),
  'abs':(np.abs,np.sign,[0.]),
  'softplus':(lambda z:np.logaddexp(0.,z),expit,[]),
  'silu':(lambda z:np.asarray(z)*expit(z),_silu_prime,[]),
  'gelu':(lambda z:np.asarray(z)*ndtr(z),_gelu_prime,[]),
  'sine':(np.sin,np.cos,[]),'tanh':(np.tanh,_tanh_prime,[]),
  'erf':(erf,lambda z:2/math.sqrt(math.pi)*np.exp(-np.asarray(z)**2),[]),
  'gaussian_rbf':(lambda z:np.exp(-.5*np.asarray(z)**2),lambda z:-np.asarray(z)*np.exp(-.5*np.asarray(z)**2),[]),
 }
 if name not in table:raise ValueError(f'Unknown canonical link {name!r}')
 return table[name]


def integrate_gaussian(function):
 """Independent scalar reference on the full real line, with fixed tolerances."""
 cuts=(-math.inf,-6.,-4.,-2.,0.,2.,4.,6.,math.inf)
 values=[];errors=[]
 for left,right in zip(cuts[:-1],cuts[1:]):
  value,error=quad(lambda z:float(function(z))*math.exp(-.5*z*z)/SQRT2PI,left,right,
                   epsabs=2e-14,epsrel=2e-13,limit=200)
  values.append(value);errors.append(error)
 return math.fsum(values),math.fsum(errors)


@dataclass(frozen=True)
class Link:
 name:str
 terms:tuple
 function:object
 derivative:object
 breaks:tuple
 gaussian_moments:tuple
 derivative_second:float
 raw_zq:float
 moment_source:str
 moment_error_estimate:tuple
 def __call__(self,z):return self.function(z)
 def metadata(self):
  return dict(name=self.name,terms=self.terms,definition=DEFINITIONS.get(self.name,'declared linear combination of raw canonical links'),
   kinks=list(self.breaks),raw_mean=self.gaussian_moments[0],raw_second=self.gaussian_moments[1],
   derivative_second=self.derivative_second,raw_zq=self.raw_zq,moment_source=self.moment_source,
   moment_error_estimate=self.moment_error_estimate,
   normalization='Raw mean retained; additive target normalized to E[Y^2]=1; no centering')


@lru_cache(maxsize=128)
def make_link(spec):
 """Canonical pure links or explicit sums such as h3+0.1*h5; no products."""
 if not isinstance(spec,str) or not spec.strip():raise ValueError('Nonempty scalar link specification required')
 terms=[]
 for part in spec.replace(' ','').split('+'):
  pieces=part.split('*')
  if len(pieces)==1:weight,name=1.,pieces[0]
  elif len(pieces)==2:weight,name=float(pieces[0]),pieces[1]
  else:raise ValueError(f'Invalid scalar link term {part!r}')
  name=ALIASES.get(name,name)
  if name not in CANONICAL_NAMES or not math.isfinite(weight):raise ValueError(f'Unknown/nonfinite link term {part!r}')
  terms.append((weight,name))
 primitives=[(w,*primitive(n)) for w,n in terms]
 function=lambda z:sum(w*f(z) for w,f,_,_ in primitives)
 derivative=lambda z:sum(w*df(z) for w,_,df,_ in primitives)
 breaks=tuple(sorted({b for _,_,_,br in primitives for b in br}))
 name=terms[0][1] if len(terms)==1 and terms[0][0]==1 else spec
 if name in CLOSED_MOMENTS:
  mu,m2,d2=CLOSED_MOMENTS[name];source='closed Gaussian formula';errors=(0.,0.,0.)
 else:
  mu,e1=integrate_gaussian(function);m2,e2=integrate_gaussian(lambda z:function(z)**2)
  d2,e3=integrate_gaussian(lambda z:derivative(z)**2)
  if name=='tanh':mu=0.
  source='fixed independent full-line scalar quadrature';errors=(e1,e2,e3)
 if not (math.isfinite(mu) and math.isfinite(m2) and math.isfinite(d2) and m2>0 and m2>=mu*mu):
  raise ValueError('Nonfinite/zero scalar link second moment')
 zq_closed={**{f'h{k}':0. for k in (2,3,4,5)},'relu':.5,'leaky_relu':.55,'abs':0.,
             'softplus':.5,'silu':.5,'gelu':.5,'sine':math.exp(-.5),
             'tanh':1-m2,'erf':2/math.sqrt(3*math.pi),'gaussian_rbf':0.}
 zq=zq_closed[name] if name in zq_closed else integrate_gaussian(lambda z:z*function(z))[0]
 link=Link(name,tuple(terms),function,derivative,breaks,(mu,m2),d2,zq,source,errors)
 return link,list(breaks)
