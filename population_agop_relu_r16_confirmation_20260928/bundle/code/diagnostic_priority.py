"""Loss-prefix-only diagnostic scheduling; no training or feature dependence."""

def diagnostic_order(saved_steps, observed, endpoint, strict_exit, fractions=None):
    states=set(saved_steps)
    # Preserve the historical boundary order exactly, including absent states.
    boundaries=[0,strict_exit-1 if strict_exit is not None else -1,
                strict_exit if strict_exit is not None else -1,
                endpoint,endpoint+1,observed]
    interior=[]
    if fractions:
        strict_endpoint=strict_exit-1 if strict_exit is not None else observed
        for end in [strict_endpoint,endpoint]:
            available=sorted(s for s in states if 0<=s<=end)
            if not available:continue
            for fraction in fractions:
                target=fraction*end
                # The smaller checkpoint wins an exact distance tie.
                interior.append(min(available,key=lambda s:(abs(s-target),s)))
    return list(dict.fromkeys(s for s in boundaries+interior+sorted(states) if s in states))
