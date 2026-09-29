# Batched construction of the same integration rule

Prepared prototype; not launched or validated as a model implementation.

The reference transition integration resolves the gate transitions, but its full-width evaluation spends substantial time constructing many small conditional integration rules in Python. This separate fork batches those constructions. It uses the same scalar Gaussian nodes, weights, order, truncation, moment formulas, teacher, parameter state, and output normalization. The original implementation remains in student_transition.py and is not edited.

- transition_batch.py and its focused scalar tests are owned by the performance reviewer.
- student_transition_fast.py is an opt-in subclass that replaces only the inner conditional-node construction inside pair_terms. All diagonal and teacher methods are inherited from the unchanged reference.
- probe.py retains the same original64-neuron fixed states, orders12/16 and22evaluations as transition_fullwidth. Its soft cap is590seconds; dispatch must supply TERM595/KILL600. It uses one CPU and makes no training updates.
- Input/source manifest and independent review are required before execution.

Scalar node equivalence does not alone certify model outputs or gradients. After scalar and source review, the bounded server probe must be compared against every completed reference base state and finite-difference value, preserving reference censoring. Report full losses, full gradients, Gram/t arrays, symmetry, order and tail sensitivity, and runtime. Passing fixed-state checks still does not establish a corrected training trajectory or a new learning result.

No numerical tolerance, node resolution or mathematical objective is reduced for speed. Further optimization requires its own evidence. Local work is limited to scalar mathematics, source compilation and saved-array analysis; model evaluations remain server-only.

