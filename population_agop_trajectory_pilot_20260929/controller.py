"""Model-independent safeguarded Euler controller and loss-only event bookkeeping.

Imports only NumPy/stdlib. Synthetic tests never import an evaluator.
"""
import hashlib
import math
import numpy as np


class NumericalFailure(RuntimeError):
    pass


class NumericalStall(RuntimeError):
    pass


def norm(parts):
    return float(np.sqrt(sum(np.sum(x*x) for x in parts)))


def digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def state_hash(theta):
    return hashlib.sha256(''.join(digest(x) for x in theta).encode()).hexdigest()


def initialize(cfg):
    """Original draw order and multiplication/division order, including heads."""
    rng = np.random.default_rng(cfg['seed'])
    raw = (rng.standard_normal((cfg['d']+1, cfg['m'])),
           rng.standard_normal((cfg['d']+1, cfg['m'])), rng.standard_normal(cfg['m']))
    theta = (raw[0]*cfg['s']/math.sqrt(cfg['d']+1),
             raw[1]*cfg['s']/math.sqrt(cfg['d']+1),
             raw[2]*cfg['s']*cfg['head_ratio'])
    return theta, dict(gaussian_hashes=[digest(x) for x in raw],
                      parameter_hashes=[digest(x) for x in theta], state_hash=state_hash(theta))


def directions(ev, cfg):
    # Byte-for-byte arithmetic order of frozen update_directions.
    P, V, a = cfg['m']*ev['gP'], cfg['m']*ev['gV'], cfg['m']*ev['ga']
    if cfg['head_lr'] != 1.: a = cfg['head_lr']*a
    return P, V, a


def proposal(theta, ev, cfg, t):
    D = directions(ev, cfg)
    gn = np.sqrt((D[0]**2).sum(0)+(D[1]**2).sum(0)+D[2]**2)
    th = np.sqrt((theta[0]**2).sum(0)+(theta[1]**2).sum(0)+theta[2]**2)
    rel = float((gn/np.maximum(th, 1e-300)).max())
    dt = min(cfg['dt_max'], cfg['h']/max(rel, 1e-300), max(0., cfg['t_max']-t))
    slope = sum(float(np.sum(ev[k]*x)) for k, x in zip(('gP','gV','ga'), D))
    if not np.isfinite([dt, rel, slope]).all() or slope < 0:
        raise NumericalFailure('Nonfinite or non-descent proposal')
    return D, dt, rel, slope


def backtrack(theta, ev, cfg, t, controls, evaluate, record):
    """No mutation. evaluate must validate every returned trial, even rejection.

    Strict decrease is mandatory. The Armijo RHS alone admits 8 eps roundoff;
    if its requested decrease is unresolvable, stop without accepting a step.
    Maximum_backtracks=20 permits the initial trial plus at most 20 halvings.
    """
    D, dt, rel, slope = proposal(theta, ev, cfg, t)
    L = float(ev['L'])
    tol = controls['roundoff_factor']*np.finfo(float).eps*max(
        controls['roundoff_scale_floor'], abs(L), abs(float(ev['V'])))
    for trial in range(controls['maximum_backtracks']+1):
        required = controls['armijo_c1']*dt*slope
        if dt <= 0 or required <= tol:
            record(dict(trial=trial, dt=dt, status='numerical_stall',
                        required_decrease=required, roundoff_tolerance=tol, descent_slope=slope))
            raise NumericalStall('Armijo decrease below declared roundoff resolution')
        candidate = tuple(x-dt*d for x,d in zip(theta,D))
        trial_ev = evaluate(candidate, trial)
        trial_L = float(trial_ev['L'])
        if not np.isfinite(trial_L):
            raise NumericalFailure('Nonfinite trial loss')
        accepted = bool(trial_L < L and trial_L <= L-required+tol)
        record(dict(trial=trial, dt=dt, L=trial_L, base_L=L, rel=rel,
                    descent_slope=slope, required_decrease=required,
                    armijo_rhs=L-required, roundoff_tolerance=tol,
                    state_hash=state_hash(candidate), status='accepted' if accepted else 'rejected'))
        if accepted:
            return candidate, trial_ev, dt, rel
        dt *= controls['backtrack_factor']
    raise NumericalFailure('Armijo backtracking exhausted; no update accepted')


class Checkpoints:
    def __init__(self, cfg):
        self.cfg=cfg; self.next_cp=0.; self.last_dL=1e-8; self.last_L=1.

    def natural(self,t,L):
        dL=1.-L
        yes=(t>=self.next_cp or (dL>1e-8 and dL>self.cfg['dl_ratio']*self.last_dL
             and L>self.cfg['L_stop']) or abs(L-self.last_L)>.02)
        if yes:
            self.last_dL=max(dL,1e-8); self.last_L=L
            self.next_cp=max(t*self.cfg['cp_ratio'],t+self.cfg['cp_min'])
        return yes


class Events:
    """Exact all-accepted-update prefixes; diagnostics never affect updates."""
    def __init__(self, criteria):
        self.criteria=criteria; self.low=float('inf'); self.high=-float('inf')
        self.initial=None; self.exits={}; self.candidates={}; self.releases={}

    def observe(self, step, t, L, metrics, screen):
        if self.initial is None: self.initial=(metrics,screen)
        self.low=min(self.low,L); self.high=max(self.high,L)
        roles=[]; initial, initial_screen=self.initial
        for ratio in self.criteria['initial_loss_ratio_tolerances']:
            key=str(ratio)
            if key not in self.exits and (L<=0 or not np.isfinite(L) or self.high>ratio*self.low):
                self.exits[key]=dict(first_exit_step=step,first_exit_t=t,last_valid_step=step-1)
                roles.append('prefix_crossing:'+key)
            stable=initial_screen['refit_stable'] and screen['refit_stable']
            resolved=initial_screen['agop_resolved'] and screen['agop_resolved']
            if key not in self.exits and key not in self.candidates and stable and resolved:
                first=[x['actual_mse'] for x in initial['pinv'].values()]
                now=[x['actual_mse'] for x in metrics['pinv'].values()]
                gain=min(first)-max(0.,max(now))
                amin=metrics['agop_Amin']-initial['agop_Amin']
                if gain>=self.criteria['refit_gain_over_variance_min'] and amin>=self.criteria['delta_Amin_min']:
                    self.candidates[key]=dict(step=step,t=t,L=L,delta_Amin=amin,refit_gain=gain)
                    roles.append('first_candidate:'+key)
            if key in self.candidates and key not in self.releases:
                c=self.candidates[key]
                if step>c['step'] and c['L']-L>=self.criteria['later_raw_loss_drop_over_variance_min']:
                    self.releases[key]=dict(step=step,t=t,L=L,drop_over_variance=c['L']-L)
                    roles.append('first_release:'+key)
        return roles

    def summary(self):
        return dict(loss_low=self.low,loss_high=self.high,exits=self.exits,
                    candidates=self.candidates,releases=self.releases,
                    claims='Observed numerical sequences; sampled audits are not trajectory certificates')
