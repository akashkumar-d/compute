"""Model-independent bounded Armijo step; importing this module evaluates no model."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Settings:
    c1: float = 1e-4
    shrink: float = 0.5
    max_trials: int = 25

    def __post_init__(self):
        for name in ('c1', 'shrink'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(name + ' must be finite and strictly between zero and one')
        if isinstance(self.max_trials, bool) or not isinstance(self.max_trials, int) or not 1 <= self.max_trials <= 64:
            raise ValueError('max_trials must be an integer in [1,64]')


@dataclass
class Result:
    state: object
    evaluation: object
    accepted_dt: float
    reason: str
    trials: list

    @property
    def accepted(self):
        return self.reason == 'accepted'


def raw_metric_slope(squared_norms, m, head_lr, variance):
    """Nominal slope from quadrature gradients for -m(gP,gV,head_lr*ga)."""
    if len(squared_norms) != 3 or any(not math.isfinite(x) or x < 0 for x in squared_norms):
        return float('nan')
    if any(not math.isfinite(x) or x <= 0 for x in (m, head_lr, variance)):
        return float('nan')
    gp2, gv2, ga2 = squared_norms
    return -m * (gp2 + gv2 + head_lr * ga2) / variance


def take_step(state, evaluation, *, dt, time, slope, make_trial, evaluate,
              loss, valid_evaluation, valid_state, same_state,
              stop=lambda: False, record=lambda event: None, settings=Settings()):
    """Never mutate the base state/evaluation; return the accepted full evaluation.

    Callbacks must be pure with respect to the base state and evaluation. The
    caller owns deadlines; stop is checked both before and after each trial
    evaluation. A running callback cannot be preempted here. Only a finite,
    nonnegative objective and finite gradients may be accepted. No numerical
    slack permits an increase or equality. The step count/time belong to the
    caller and advance only when Result.accepted is true.
    """
    trials = []
    base_loss = float(loss(evaluation))

    def finish(reason):
        return Result(state, evaluation, 0.0, reason, trials)

    def emit(index, amount, value, target, reason, error=None):
        event = dict(trial=index, dt=amount, loss_before=base_loss,
                     loss_trial=value if value is not None and math.isfinite(value) else None,
                     armijo_target=target if math.isfinite(target) else None,
                     nominal_normalized_directional_slope=slope, accepted=reason == 'accepted',
                     reason=reason, error=error)
        trials.append(event)
        record(event.copy())

    if not math.isfinite(base_loss) or base_loss < 0 or not valid_evaluation(evaluation):
        return finish('invalid_base_evaluation')
    if not math.isfinite(slope) or slope > 0:
        return finish('invalid_descent_direction')
    if slope == 0:
        return finish('stationary_or_underflowed_direction')
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(time):
        return finish('invalid_proposed_step')
    for index in range(settings.max_trials):
        if stop(): return finish('external_stop_before_trial')
        if dt <= 0 or time + dt == time or not math.isfinite(time + dt):
            return finish('time_step_stagnation')
        target = base_loss + settings.c1 * dt * slope
        if target == base_loss:
            return finish('armijo_decrease_unresolvable')
        candidate = make_trial(state, dt)
        if same_state(state, candidate):
            emit(index, dt, None, target, 'state_stagnation')
            return finish('state_stagnation')
        if not valid_state(candidate):
            emit(index, dt, None, target, 'nonfinite_trial_state')
            dt *= settings.shrink
            continue
        try:
            trial_ev = evaluate(candidate)
            trial_loss = float(loss(trial_ev))
            valid = math.isfinite(trial_loss) and trial_loss >= 0 and valid_evaluation(trial_ev)
        except Exception as exc:
            emit(index, dt, None, target, 'evaluation_exception', repr(exc))
            return finish('evaluation_exception')
        if stop():
            emit(index, dt, trial_loss, target, 'external_stop_after_evaluation')
            return finish('external_stop_after_evaluation')
        accept = valid and trial_loss < base_loss and trial_loss <= target
        reason = 'accepted' if accept else 'invalid_trial_evaluation' if not valid else 'armijo_rejected'
        emit(index, dt, trial_loss, target, reason)
        if accept:
            return Result(candidate, trial_ev, dt, 'accepted', trials)
        dt *= settings.shrink
    return finish('line_search_exhausted')
