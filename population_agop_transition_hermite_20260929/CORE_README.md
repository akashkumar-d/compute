# Hermite pair-moment prototype

Diagnostic only: no training has used this code. Compute per-neuron coefficients
`c[j,r,n] = E[S^(r)(bias_j + scale_j Z) h_n(Z)]` for derivatives0–4, then Gaussian
pair kernels by summing `c[j,r,n] c[k,s,n] rho[j,k]^n`. Affine-value factors follow
Gaussian integration by parts using actual input-vector covariances. All ten
ordered moment fields retain their original orientations.

The inherited diagonal and pure-h3 teacher formulas are unchanged. This replaces
pair quadrature with a truncated Hermite series; it is not an exact finite-series
loss gradient or exact Gaussian integration. No adaptive masks or training
interface have been validated. Tail estimates use finite-quadrature residual
energies, whose raw signed values and flags remain available. They are not error
certificates. Nonzero scale underflow fails explicitly; exact zero is handled
deterministically. Correlation overshoot within64 machine epsilons is clipped and
flagged, while larger overshoot fails. These are actual marginal covariances,
not the reference conditional-basis rounding/tiny-residual-drop approximation.

The author reports ten synthetic tests passing in1.073seconds before a local
disk-space error blocked this final note. They cover direct polynomial Gaussian
integration of all ten fields, zero scales, correlation signs, row selection,
energy-error propagation, invalid inputs and a fake base-class interface, with
no population-model import or evaluation. Core hash:
`8bc2f640c3a91daa6f488471902618b71f5c34c28f635862220ee2c4ece6f7e6`.
Root changed only the test's scalar-rule lookup to the byte-identical same-directory
copy for a standalone bundle; that path adjustment needs its own verification.

The tests expose a limitation: degree128/order16 has spurious high-degree
coefficients as large as6.28e-9 even for a cubic polynomial, despite valid energy
flags. Order24 reduces these to about5.62e-14. Energy flags alone cannot establish
accuracy. No model-state runtime or accuracy claim is made by these tests.

See `PROBE_PROTOCOL.md` for the fixed, prospective server comparison. All results,
including failed base screens, must be retained; no automatic training follows.
