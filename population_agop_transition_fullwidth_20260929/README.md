# Full-width fixed-state numerical validation

Prepared for the original pure-cubic SwiGLU run: r=8, d=64, m=64, seed 641, initial inner scale 0.3, head ratio 0.3 and head rate 0.0009. No neurons or biases are removed, the original alpha is retained, and all original coefficients and teacher normalization are preserved.

This is the necessary next accuracy check after the separate four-neuron probe. It is not a training experiment and does not claim that a learning criterion is met.

## Fixed protocol

Use the original saved step 472 (last saved valid 5% initial-window state) and step 2260 (before the old bad update). At each state:

1. Evaluate the complete loss and gradient at quadrature order 12.
2. Form the original m=64/head-rate weighted gradient, then normalize that direction once. Compare its nominal negative directional derivative with central differences at 1e-4 and 1e-5 times max(1, Euclidean parameter norm).
3. Repeat at order 16, using its own base direction.
4. Compare order 12/16 loss, Gram matrix, teacher correlations and full gradients.
5. Compare the order 16 base at Gaussian limits 12 and 10.

This gives 22 evaluations and four order slots. Per-call runtimes, Gram/p11 asymmetry, dropped numerical gate residuals and unchanged-input checks are recorded. Preserve every missing or capped slot.

## Execution bounds

Server only: one single-thread CPU on the authorized 32-core Studio; reduced priority; no GPU. Soft deadline 890 seconds, external TERM at 895 seconds and hard kill at 900 seconds. Local work is source review or saved-data analysis only.

The caller must check the exact manifest review, publish to the shared compute branch, check live capacity and allowance, and use the required external timeout. Never launch at or below 2% weekly remaining. Do not overwrite an existing run.

## Interpretation

Passing directional checks (relative discrepancy <=0.001) supports numerical consistency at these two fixed states. It is not a whole-trajectory, AGOP, refit or Gaussian-tail certificate, and does not upgrade prior failed or censored learning experiments. Order convergence and runtime determine whether this reference integration can support a later bounded training test. The broader scientific goal and all original criteria remain unchanged.

Source parent: transition_quadrature, published 21d7170. Original model kernels are byte-identical; only the probe's scope, integration orders, call count and runtime limits change. PROBE_MANIFEST.json records the exact pins.

