# Rank sweep: fixed raw teachers and matched initialization

**Design only; no runs or model evaluations were performed to prepare this protocol.** First batch: 32 arms, h2 and raw ReLU teachers × ReLU/SwiGLU students × ranks2/4/8/16 × reused development seeds641/642. Dimension stays64; widths stay256 for ReLU and64 for SwiGLU. [DESIGN.json](DESIGN.json) contains every exact proposed configuration, historical template, explicit configuration difference, source hash, and rank-dependent teacher energy.

H2 supplies a strong pure nonlinear reference. Raw ReLU supplies a nonzero-mean, substantially linear teacher whose rank8 SwiGLU runs learned all directions only after the initial plateau ended. This pair distinguishes a rank effect from the already-observed weak-link timing problem without immediately repeating all14 links. H3 is deferred pending the parent quadrature review. Raw softplus and sine are the next small block because they add, respectively, a small nonlinear residual and an odd nonlinear residual; they are not included in the first32 arms.

## Teacher normalization and the estimand

For each canonical raw link q, let μ=E[q(Z)], σ²=Var(q(Z)), β=E[Zq(Z)] and η=E[q′(Z)²], using the existing frozen scalar moments. At rank r use nested axes U_r=(e₁,…,e_r), all additive coefficients equal to1, and

Y_r = γ_r Σᵢ₌₁ʳ q(Xᵢ), γ_r=[rσ²+r²μ²]⁻¹ᐟ².

Thus E[Y_r²]=1, E[Y_r]=rγ_rμ, and Var(Y_r)=σ²/(σ²+rμ²). This is exactly the existing uncentered teacher convention for both students. Profile the output intercept in every run. Do not center q, remove its linear term, or change normalization after outcomes. ReLU's internal coefficients1/√r divided by its scalar norm are algebraically identical to SwiGLU's cᵢ=1 followed by γ_r.

The rank comparison therefore holds the **raw link and second-moment convention** fixed. It does not hold raw target variance or raw gradient amplitude fixed when μ≠0. Normalizing reported losses by Var(Y_r) does not rescale the objective used for training. A separate variance-one teacher experiment could address another question, but is not part of this comparison and cannot replace a failed primary rank cell.

| Rank | h2 γ_r | h2 Var(Y) | Raw-ReLU γ_r | Raw-ReLU Var(Y) | Raw-ReLU stop MSE |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.707107 | 1 | 0.870946 | 0.517094 | 0.00517094 |
| 4 | 0.5 | 1 | 0.505731 | 0.348703 | 0.00348703 |
| 8 | 0.353553 | 1 | 0.278286 | 0.211169 | 0.00211169 |
| 16 | 0.25 | 1 | 0.147127 | 0.118049 | 0.00118049 |

Every arm stops at the same **normalized** loss0.01 if reached. The ReLU runner accepts raw MSE, so its configuration stores 0.01×Var(Y_r); SwiGLU already accepts MSE/Var(Y). This is a predeclared stopping change from v7, not a change to the success criterion. New rank8 controls use these same settings; old rank8 data provide historical context only.

## Per-direction signal and baseline geometry

Each coordinate contributes centered target energy Var(Y_r)/r, and nonlinear energy γ_r²(σ²−β²). The total linear and nonlinear fractions of target variance are β²/σ² and 1−β²/σ², independent of r. Their **per-direction** fractions shrink as1/r. With μ=0 the coefficient γ_r scales as1/√r; with a nonzero mean it eventually scales as1/r, and per-direction raw energy decays as1/r². Thus a common physical horizon may be much harder at high rank under the preserved raw normalization.

The teacher's own population AGOP also illustrates the distinction. Inside its r-dimensional support its aggregate-direction eigenvalue is γ_r²[η+(r−1)β²], and its r−1 transverse eigenvalues are γ_r²(η−β²); outside the support the eigenvalues are zero. These are teacher properties, **not predictions for the student's AGOP**. H2 has r equal eigenvalues2/r. Raw ReLU has transverse eigenvalueγ_r²/4 and aggregate/transverse ratio r+1:

| Rank | h2 teacher weakest eigenvalue | Raw-ReLU teacher weakest eigenvalue | Raw-ReLU aggregate/weakest ratio |
|---:|---:|---:|---:|
| 2 | 1 | 0.189637 | 3 |
| 4 | 0.5 | 0.0639409 | 5 |
| 8 | 0.25 | 0.0193608 | 9 |
| 16 | 0.125 | 0.00541159 | 17 |

This growing anisotropy gives a scientific reason to inspect the weakest direction separately from top-direction alignment as rank rises. It does not establish a numerical failure at rank16.

| Rank | Random-subspace mean alignment benchmark r/d | ReLU width/r | SwiGLU width/r | ReLU refit l1 cap |
|---:|---:|---:|---:|---:|
| 2 | 0.03125 | 128 | 32 | 90.5097 |
| 4 | 0.0625 | 64 | 16 | 128 |
| 8 | 0.125 | 32 | 8 | 181.019 |
| 16 | 0.25 | 16 | 4 | 256 |

For an isotropically distributed rank-r student eigenspace independent of a fixed rank-r teacher subspace, expected **mean** squared-principal-cosine alignment is r/d. Expected projection of a random top direction into the teacher subspace is also r/d. **Minimum alignment is not r/d**; its distribution changes with r, and the primary statistic must subtract the actual measured initialization A_min(0) in each arm. Use the original ≥0.5 gain threshold. If a particular initialization leaves less than0.5 possible gain, record that effect-size limitation rather than changing the threshold. With2r≤d here, intersection geometry forces no nonzero common subspace.

Fixed width means width per teacher direction decreases with r; this is part of the stated finite-width comparison, not an isolated infinite-width rank law. The ReLU comparator keeps its inherited l2≤32, l1≤64√r convention at every state. Its budget therefore changes across ranks even though it remains fixed within each trajectory. SwiGLU retains its unrestricted readout cutoff-envelope comparator. Cross-rank refit comparisons must disclose this distinction; do not silently freeze or enlarge one comparator's cap.

## Paired seeds and exact settings

Seeds641/642 are reused **paired development seeds**, not fresh confirmation. At fixed student, d, width and scale, the existing initialization code consumes the same random sequence independently of r and teacher: ReLU draws A,W,b; SwiGLU draws P,V,a. Use identical Gaussian arrays for the same student/seed in every rank/link cell, without rank-dependent seed offsets. Nested teacher axes make the low-rank teacher directions a subset of the high-rank directions. Before interpreting results, verify initial-array hashes across those cells. Source inspection establishes the intended pairing; this preparation did not generate or evaluate arrays. Matching the integer seed across architectures does not make their different-shaped arrays identical.

| Setting | ReLU student | SwiGLU student |
|---|---|---|
| d / width | 64 /256 | 64 /64 |
| Initialization | v7 scale0.01; coordinates0.01/√64 | v7 scale0.1; P,V coordinates0.1/√65; head ratio1 |
| Trained blocks | A,W,b plus profiled intercept | P,V,a plus profiled intercept |
| Optimizer | existing joint Armijo population GD | existing block-rate adaptive Euler |
| Step controls | h=0.5, existing line search | h=0.01, dt_max=50 |
| Head rate | existing joint recipe | **0.01 for both h2 and raw ReLU** |
| Horizon / update cap | force_time_max1500 / steps20000 | t_max3000 / max_steps20000 |
| Loss stop | raw MSE=0.01 Var(Y_r) | normalized MSE=0.01 |
| Saved grid | stride25 plus inherited geometric/boundary snapshots | cp_ratio1.06, cp_min0.5, dl_ratio1.5 |
| Diagnostic priority | existing initial/boundary/terminal plus fractions0.25/0.5/0.75 | same declared fraction policy |
| Numerical recipe | existing analytic h2/raw-ReLU teacher cross kernels and angular student kernel | **pair32, diagonal96, conditional48, teacher24 for every rank** |

The head0.001 quadratic trajectory being validated separately is not the head0.01 rank study. Reusing its higher-order findings informs numerical precautions but does not validate this sweep. The longer SwiGLU horizon/lower stop use the existing v8-style observation settings; the planned configuration explicitly has20000 maximum updates. There is no implied common clock between students or exact speed scaling across ranks.

Configuration lineage: each planned arm clones the same-student/teacher/seed v7 rank8 JSON under `goal_followup_v7/breadth14/cpu32/bundle/configs/`. ReLU changes ID, rank and variance-dependent loss stop. SwiGLU changes tag/cell/rank, c-list length, common quadrature orders, time/update limits, and loss stop. All other science fields are identical to those templates. The exact objects and differences are embedded in DESIGN.json; no executable configurations or launcher were modified by this design task.

## Rank16 feasibility and numerical limits

Rank16 is **dimensionally valid**, not numerically certified. The existing SwiGLU validator requires rank=len(c)≥1, d≥2r and m≥r; 64≥32 and64≥16 satisfy those conditions. ReLU's teacher support also fits in d64, with ample width256 relative to16. No width or dimension restriction forces exclusion. This is not a claim that finite width64 guarantees every canonical teacher can be accurately represented or learned.

The numerical integration routines exploit additive teacher coordinates; they do not form an r-dimensional tensor quadrature. Raising r adds teacher-coordinate contributions while student pair work remains tied to width. Runtime can still grow, and weak eigenvalue gaps/conditioning can worsen. Orders32/96/48/24 are a common **provisional** recipe, not evidence that rank16 is accurate or that order2 eliminates all rank8 issues.

Retain the original relative AGOP eigenvalue/gap guard1e-9 and refit tolerance1e-7. For each rank, prioritize higher-order saved-state checks at initialization, both exact initial-loss boundaries, the first putative same-state candidate, and terminal state. For SwiGLU compare against64/192/96/48, including loss, **actual block-weighted update**, adaptive dt, AGOP eigenspace and refit sensitivity. A raw concatenated gradient comparison alone is inadequate when block rates differ. Do not promote an unresolved candidate or interpret a quadrature-dependent stall as an impossibility result. If refinement invalidates a rank, retain its original record and prepare a separate matched precision comparison across affected ranks; do not mix orders silently within a rank curve or omit rank16 because it is inconvenient.

Order checks, spectral guards and cutoff agreement remain numerical evidence, not formal integration or oracle-risk certificates. The pre-existing student-specific refit comparators remain different.

## Unchanged success criteria and bounded execution plan

For each rank/teacher/student/seed report the initial whole-update max/min loss prefixes≤1.01 and≤1.05 separately. At one saved checkpoint in that prefix require ΔA_min≥0.5 and refit improvement≥0.1 Var(Y_r), with initial/current numerical screens. Then require a subsequent same-run loss decrease≥0.1 Var(Y_r). Preserve signed gains, all missing diagnostics, exact first crossing updates, caps and numerical flags. Aggregate with the canonical summary criterion (the inherited `summarize_coverage.py` rule); the native ReLU runner still carries a legacy raw-refit `>1e-5` event flag, which must not be used as the primary `.1 Var(Y_r)` qualification. No source change to that legacy flag is needed for this design. Late windows and eventual alignment remain separate outcomes. A refit already below0.1 Var at initialization must be marked potentially effect-size-ineligible under the saved numerical evidence; the affine residual alone cannot establish that for the actual random representation.

The parent has verified a32-CPU Studio and is running two independent numerical-validation workers. The rank batch may use **at most28 single-thread workers**, leaving two CPUs for that validation and two free. The rank launcher should therefore reserve4 relative to its own pool, and the combined workstreams must stay at most30. Per arm:1800seconds training +180diagnostics +10cleanup=1990seconds. Global rank-batch ceiling3600seconds plus separate setup120seconds permits at most two waves, but is shorter than two fully funded1990-second arms. Later starts may therefore be cut short by the global deadline. These are parent execution instructions, not a launcher change made here; this task performed no network operation or launch.

Keep the frozen queue in DESIGN.json: ranks16,8,4,2 in descending order to start the more demanding cells early; within each rank seed641 then642; within each seed teachersh2/raw ReLU; within each teacher ReLU/SwiGLU students. Architecture, teacher and seed ordering is deterministic and independent of outcomes. Different completion times do not change arm inclusion. The budgets grant observation time, not a promise to reach the configured horizon. An unstarted, pending, failed, wall-capped or diagnostically incomplete arm remains explicit. Late rank2 starts may receive less observation time under the global ceiling; compare common observed support and retain that rank-dependent censoring, without treating the missing horizon as a failure. Exact rank8 controls are new arms at the same precision/limits; do not pool old base-order rank8 traces as matched controls.

## Expansion after the first batch

The complete14-link design would contain224 arms. The first32 leave192 additional arms: twelve teachers ×two students ×four ranks ×two seeds. Do not launch that entire expansion merely because rank16 passes structural validation.

Read and numerically audit the first batch, then add a small reviewed block: raw softplus and sine first, preserving all four ranks, both students, two seeds and fixed criteria. H3 remains pending the current quadrature review; h4/h5 need explicit horizon-censoring and small-signal treatment. The other canonical links can follow in disclosed waves using the exact same raw-moment formula; DESIGN.json already records energy scaling for all14 links at all four ranks. Expansion is a readiness and diagnostic decision, not permission to drop unsuccessful ranks, center weak teachers, or retune a rank until it passes. Any recipe change requires a separately labeled matched comparison. Fresh-seed confirmation follows only after a fixed recipe survives development and numerical checks.

Relevant sources: [v7 protocol](../../goal_followup_v7/breadth14/cpu32/bundle/PROTOCOL.md), [frozen teacher moments](../../goal_followup_v7/breadth14/cpu32/bundle/TEACHERS.json), [SwiGLU validation and runner](../../goal_followup_v7/breadth14/cpu32/bundle/swiglu/code/run_one.py), [SwiGLU normalization](../../goal_followup_v7/breadth14/cpu32/bundle/swiglu/code/engine/swpop.py), [ReLU runner](../../goal_followup_v7/breadth14/cpu32/bundle/code/scaling_run.py), [ReLU refit budgets](../../goal_followup_v7/breadth14/cpu32/bundle/code/refit.py), and [weak-teacher saved-data triage](../weak_teacher_triage/TRIAGE.md). No manuscripts, prior datasets, launchers or scientific source files were changed.
