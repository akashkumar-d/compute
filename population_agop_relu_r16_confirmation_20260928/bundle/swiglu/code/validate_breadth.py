"""Bounded single-thread scalar/evaluator validation; no parameter updates/training."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import ast,hashlib,importlib.util,json,math,signal,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/engine'))
import canonical_links as links
import swpop,swsmall,swsmall_periodic

def alarm(*_):raise TimeoutError('20-second bounded evaluator validation exceeded')
signal.signal(signal.SIGALRM,alarm);signal.alarm(20)
started=time.monotonic();rows=[];gradient_checks=[]
rng=np.random.default_rng(87321);d,m,r=6,4,2
P=rng.normal(size=(d+1,m))*.22;V=rng.normal(size=(d+1,m))*.19;a=rng.normal(size=m)*.3
parts=[rng.normal(size=p.shape) for p in (P,V,a)]
parts=[p/np.linalg.norm(p) for p in parts]
for name in links.CANONICAL_NAMES:
 link,breaks=links.make_link(name)
 # Independent full-line scalar integrals validate closed formulas as well as fixed moments.
 expected=[links.integrate_gaussian(g)[0] for g in (link,lambda z:link(z)**2,lambda z:z*link(z),lambda z:link.derivative(z)**2)]
 actual=[*link.gaussian_moments,link.raw_zq,link.derivative_second]
 np.testing.assert_allclose(expected,actual,rtol=2e-12,atol=2e-13)
 z=np.array([-3.1,-1.7,-.4,.25,.9,2.2,3.3]);eps=1e-6
 fd=(link(z+eps)-link(z-eps))/(2*eps)
 np.testing.assert_allclose(fd,link.derivative(z),rtol=3e-8,atol=2e-8)
 teachers=[swpop.Teacher(link,np.ones(8),breaks=breaks,n_x=n) for n in (6,12,24,48)]
 assert all(t.gamma==teachers[0].gamma and t.EY==teachers[0].EY and t.V==teachers[0].V for t in teachers)
 mu,m2=link.gaussian_moments
 norm=math.sqrt(m2+7*mu*mu);mean=math.sqrt(8)*mu/norm;variance=1-mean*mean
 np.testing.assert_allclose([teachers[0].gamma,teachers[0].EY,teachers[0].V],[1/(math.sqrt(8)*norm),mean,variance],rtol=1e-14,atol=1e-14)
 convergence=[dict(n_x=n,mean_error=float(np.max(abs(t.quadrature_raw_mean-mu))),second_error=float(np.max(abs(t.quadrature_raw_second-m2)))) for n,t in zip((6,12,24,48),teachers)]
 assert convergence[-1]['mean_error']<1e-11 and convergence[-1]['second_error']<1e-11
 evaluator=[]
 for nx in (12,24,48):
  teacher=swpop.Teacher(link,np.ones(r),breaks=breaks,n_x=nx)
  engine=swpop.SwiGLUPop(d,teacher,alpha=1.,intercept='refit',bias=True,n_pair=32,n_diag=64,n_z=48)
  ev=engine.evaluate(P,V,a,need_grad=True);evaluator.append((engine,ev))
 reference=evaluator[-1][1]
 production=swpop.SwiGLUPop(d,swpop.Teacher(link,np.ones(r),breaks=breaks,n_x=12),alpha=1.,
                          intercept='refit',bias=True,n_pair=16,n_diag=48,n_z=24)
 production_ev=production.evaluate(P,V,a,need_grad=True)
 production_error={key:float(np.max(abs(np.asarray(production_ev[key])-np.asarray(reference[key])))) for key in ('L','gP','gV','ga')}
 assert max(production_error.values())<1e-8
 quad_errors={str(nx):{key:float(np.max(abs(np.asarray(ev[key])-np.asarray(reference[key])))) for key in ('L','gP','gV','ga')} for nx,(_,ev) in zip((12,24,48),evaluator)}
 assert max(quad_errors['24'].values())<1e-10
 engine=evaluator[-1][0];eps=2e-5;fd_checks=[]
 for idx,key in enumerate(('gP','gV','ga')):
  plus=[P.copy(),V.copy(),a.copy()];minus=[P.copy(),V.copy(),a.copy()]
  plus[idx]+=eps*parts[idx];minus[idx]-=eps*parts[idx]
  numerical=(engine.evaluate(*plus,need_grad=False)['L']-engine.evaluate(*minus,need_grad=False)['L'])/(2*eps)
  analytic=float(np.sum(reference[key]*parts[idx]));error=abs(numerical-analytic)
  assert error<3e-9+3e-6*abs(analytic),(name,key,analytic,numerical,error)
  fd_checks.append(dict(block=key,analytic=analytic,finite_difference=numerical,absolute_error=error))
 # Both diagnostic and training entry points must select the identical fixed target.
 for parser in (swsmall.make_link,swsmall_periodic.make_link):
  other,br=parser(name);assert other.gaussian_moments==link.gaussian_moments and br==breaks
 rows.append(dict(link=name,scalar_independent_absolute_errors=[abs(x-y) for x,y in zip(expected,actual)],
  scalar_derivative_fd_max_error=float(np.max(abs(fd-link.derivative(z)))),quadrature_moments=convergence,
  fixed_normalization_across_orders=True,rank8_target_mean=mean,rank8_target_variance=variance,
  evaluator_quadrature_errors=quad_errors,production_16_48_24_12_vs_32_64_48_48_errors=production_error,training_and_diagnostic_parser_agree=True))
 gradient_checks.append(dict(link=name,raw_profiled_loss=reference['L'],checks=fd_checks))
# Legacy unannotated callables retain the exact original moment arithmetic.
old_root=ROOT.parents[2]/'goal_followup_v6/swiglu_head_rate_v1/swiglu/code'
spec=importlib.util.spec_from_file_location('old_swpop',old_root/'engine/swpop.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
f=lambda z:(z*z-1)/math.sqrt(2)
before=old.Teacher(f,np.ones(2),n_x=12);after=swpop.Teacher(f,np.ones(2),n_x=12)
assert all(getattr(before,k)==getattr(after,k) for k in ('gamma','EY','V'))
assert np.array_equal(before.phix,after.phix)
oldtext=(old_root/'engine/swpop.py').read_text();newtext=(ROOT/'code/engine/swpop.py').read_text()
assert oldtext[oldtext.index('class SwiGLUPop:'):]==newtext[newtext.index('class SwiGLUPop:'):]
def function_ast(path,name):
 tree=ast.parse(path.read_text());return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
for name in ('run','update_directions','first_ratio_crossing','validate_head_lr'):
 assert function_ast(old_root/'engine/swsmall_periodic.py',name)==function_ast(ROOT/'code/engine/swsmall_periodic.py',name)
for name in ('diagnostics.py','engine/vlab.py'):
 assert (old_root/name).read_bytes()==(ROOT/'code'/name).read_bytes()
# Mixed scalar teacher support remains explicit; products are intentionally separate extras.
mixed,_=links.make_link('h3+0.1*h5');np.testing.assert_allclose(mixed.gaussian_moments,[0,1.01],atol=1e-12)
for bad in ('unknown','h3*h2','','nan*h3'):
 try:links.make_link(bad)
 except ValueError:pass
 else:raise AssertionError('Invalid link accepted: '+bad)
signal.alarm(0)
result=dict(pass_check=True,no_training=True,parameter_updates=0,single_thread=True,wall_seconds=time.monotonic()-started,
 scalar_checks=rows,profiled_training_gradient_checks=gradient_checks,unannotated_legacy_teacher_bitwise=True,
 student_loss_gradient_agop_refit_code_unchanged=True,adaptive_updates_initialization_and_checkpoint_run_ast_unchanged=True,
 raw_moment_target_normalization_fixed=True,raw_mean_retained=True,mixed_scalar_support=True,products_not_silently_reinterpreted=True,
 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').rglob('*.py'))})
(ROOT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('pass_check','no_training','parameter_updates','wall_seconds','student_loss_gradient_agop_refit_code_unchanged')}))
