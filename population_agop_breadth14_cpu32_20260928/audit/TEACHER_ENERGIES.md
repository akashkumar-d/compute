# Teacher energy before observing the new runs

All numbers concern the normalized raw additive target at rank8. Both canonical learners profile an output intercept; initial raw loss is therefore approximately the target variance. No model or training trajectory was evaluated.

|Teacher|Variance|Linear fraction of variance|Nonlinear fraction of variance|First nonconstant degree|
|---|---:|---:|---:|---:|
|h2|1|0|1|2|
|h3|1|2.62534e-33|1|3|
|h4|1|0|1|4|
|h5|1|2.0745e-33|1|5|
|relu|0.211169|0.733471|0.266529|1|
|leaky_relu|0.267218|0.804341|0.195659|1|
|abs|0.0665978|0|1|2|
|softplus|0.0496428|0.920761|0.0792389|1|
|silu|0.478266|0.79851|0.20149|1|
|gelu|0.351885|0.723288|0.276712|1|
|sine|1|0.850918|0.149082|1|
|tanh|1|0.93047|0.0695301|1|
|erf|1|0.913583|0.0864172|1|
|gaussian_rbf|0.0189707|0|1|2|

The linear component of an equal-weight additive teacher depends on the single direction proportional to the sum of the teacher axes. Learning that component can lower prediction error and align the top AGOP direction without recovering every teacher direction. The nonlinear components distinguish all-direction learning from that simpler effect.

The nonlinear fraction is an oracle residual after fitting an unrestricted affine predictor. It does **not** prove a ceiling on the actual finite student refit gain: its initialization may not represent that affine predictor well. The run report must use the actual initial feasible refit error. If that error is already smaller than the primary0.1Var(Y) improvement threshold, the threshold is infeasible for that run even though the representation may be useful. This is recorded separately; thresholds and teacher definitions remain fixed.

High-degree Hermite teachers can have very small early gradient signals; short observations remain horizon-censored rather than evidence of impossibility. These scalar energy calculations are explanatory and do not themselves establish any optimizer trajectory or feature-learning theorem.
