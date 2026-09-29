# Campaign C: preregistration

Written 2026-09-29 by `claude/plateau-review` (session_011vNRtJNdeKePn3Gex5aRg7) before any campaign-C run starts. The design, the
endpoints and the analysis below are fixed. The code that implements them is registered by its SHA-256 hashes (section 10).

## 1. Question and claim

**Question.** With every layer trained and fixed hyperparameters, does a ReLU network trained with cross-entropy on modular
addition show a *natural* training-loss plateau during which the hidden representation gains Fourier structure, followed by
generalization? "Natural" means no loss rescaling, no freezing and no controller.

The development runs (README v1.2, `natural_plateau_v1/`) say yes at depths 3–5, provided one number is set: the shrink rate
η_L·λ_L of the last hidden layer. The registered clock fixes it: x = η_L·λ_L·t₁₀, where t₁₀ is the step at which held-out
accuracy first reaches 10%. The recipe is a pilot run at a fixed rate r₀, then the rate that puts x on a target x\*.

**Primary claim, tested separately at depths 3, 4 and 5** (width 256, p = 31). For fresh seeds, the recipe produces runs that
pass the primary endpoint (section 5) at a rate whose 95% Wilson lower bound is at least 0.80. With 40 seeds per depth this means
**at least 37 of 40 runs pass** (37/40 has a lower bound of 0.801; the protocol text's "38/40" is the more conservative rounding).
The statement "at depths 3–5" is made only if all three depths pass.

## 2. Setting (unchanged from development)

- **Task.** Modular addition (a + b) mod p, all p² pairs, one-hot inputs [e_a; e_b], 80% training split.
- **Network.** Bias-free dense ReLU MLP of depth D and width 256 (512 in C4). The head is linear.
- **Hidden-layer updates.** W ← W + η·(polar(−∇W)/√rank − λ·W), where polar is the support-Polar direction. C5 replaces it with the
  Frobenius-normalized gradient.
- **Head updates.** Normalized gradient with η_V = 0.03 and λ_V = 0.2.
- **Initialization.** Scale c = 24; the deep layers use 8× He.
- **Layer 1.** η = 0.03, λ = 0.015, so the reference shrink rate is K₁ = η·λ = 4.5·10⁻⁴.
- **Deep layers.** Decay 0.07. Layers 2 … D−1 shrink at K₁; the last hidden layer shrinks at r·K₁.
- **Logging.** Loss is logged every step and observations every 20 steps. The heavy observers run every 500 steps: the legacy set
  plus the protocol-v1 set (`observers_v1`).
- **Code.** `natplat_campaign.py` reproduces the development runner `code/natplat_server.py` bit for bit when the new keys are
  absent (`tests/test_runner.py`).

## 3. Design

| stage | cohort | seeds i (init 1000+i, split 2000+i) | pilot | r₀ | x\* | steps | chains |
|---|---|---|---|---|---|---|---|
| C1 | D3, D4, D5 (width 256, p = 31) | 21–40 | short (stops at t₁₀) | 0.50 / 0.35 / 0.35 | 3.85 | 80k / 100k / 80k | 60 |
| C2 | D3, D4, D5 | 1–20 | full (horizon; the fixed-rate arm) | same | 3.85 | same | 60 |
| C3 | D4, **p = 23** | 41–60 | short | 0.35 | 3.85 | 120k | 20 |
| C4 | D4, **width 512** (2 threads per chain) | 61–64 | short | 0.40 | 4.50 | 100k | 4 |
| C5 | D3, **normalized-gradient hidden updates** | 1–20 (paired with C2 D3) | short | 0.50 | 3.85 | 80k | 20 |

Notes on the design:

- **Primary data.** The C1 and C2 recipe runs: 40 per depth.
- **Seeds.** All are disjoint from development, which used init/split seeds 2/1 and 101–105/201–205.
- **x\* = 3.85** is the value registered for the depth-5 recipe test (T23, 2026-09-28 03:35 UTC). It lies slightly below each
  depth's balance point (3.92 / 4.06 / ≈4.0 at depths 3 / 4 / 5), so the CE should stay flat or rise slightly through grokking.
- **Width 512.** These runs sit 4–9 points above the registered curve, so C4 targets x\* = 4.5.
- **Manifest.** `manifest.json` (written by `make_manifest.py`) lists every chain with:
  - its exact pilot configuration;
  - its main-run template;
  - the recipe constants;
  - the session that runs it (S1–S8, 4 cores each, longest chains first).

## 4. Procedure per chain

1. **Pilot.** Run the pilot at r₀.
   - t₁₀ is the first logged step with held-out accuracy ≥ 0.1.
   - A short pilot stops there.
   - A full pilot runs to the horizon and is also the fixed-rate arm of C2.
2. **Rate.** r = `recipe.solve_rate(t₁₀, r₀, x*)`, which solves r·K₁·t₁₀·exp(0.9·(r − r₀)) = x\*.
   - r is rounded to 0.005 and clipped to [0.10, 1.20].
   - This recipe reproduces the four T23 rates exactly.
   - If the pilot never reaches 10% (censored), t₁₀ is set to the pilot's horizon and the chain is flagged.
3. **Main run.** Run the main run at r, from the same initialization and split.
   - If r = r₀ after a full pilot, the pilot is the main run and is not repeated.
4. **Storage.** Results are stored with floats rounded to 7 significant digits. Full precision stays in the session.

## 5. Primary endpoint (one main run)

- **Knee.** The first heavy observation where the head norm ‖V‖ ≥ 0.99/λ_V.
- **Peak-band window W.**
  - Let CE(t) be the cycle-mean training loss, (L_t + L_{t+1})/2.
  - Let t\* be the step of the largest CE(t) after the knee.
  - W is the longest run of consecutive steps containing t\* on which CE(t) ≥ CE(t\*)/1.03.
  - W is computed from the training loss alone, with no reference to test accuracy.
- **Share.** The share of the rise is [clip(test(b)) − clip(test(a))]/0.8.
  - W = [a, b].
  - test(·) is held-out accuracy at the nearest logged step.
  - clip(·) clips to [0.1, 0.9].
- **A run passes** when all four hold:
  1. share ≥ **0.75**;
  2. final held-out accuracy ≥ **0.99**;
  3. A_H of the last hidden layer (class-mean energy fraction, `AH_full_h{D}`) rises by ≥ **0.02** across W (protocol F02);
  4. every logged minimum training margin in W is **> 0** (the training set stays fit).

`endpoints.py` implements this definition (`run_fields`). The analysis reports, per depth:

- k/40;
- the 95% Wilson interval;
- the number of runs failing each of the four checks.

## 6. How the endpoint behaves on the development runs (all of them run before this registration)

`python3 analyze.py dev` gives:

- **Depth 3, rule 0.5×, 4 seeds:** 4/4 pass (share 77 / 90 / 77 / 81%).
- **Depth 4, each seed at its balancing rate, 4 seeds:** 4/4 (92 / 90 / 95 / 86%). **One rate (0.35×) for all:** 2/4 (92 / 32 / 52 / 78%).
- **Depth 5, recipe (T23):** 4/4 (84 / 92 / 84 / 82%). **One rate (0.35×, T22):** 2/4 (61 / 74.9 / 90 / 77%).
- **Depth 4, width 512, 120k steps (T24):** 1/2. The tuning seed fails on final accuracy (0.85) and on fit.
- **Clock.** Of the 47 development runs at depths 3–5 (width 256, p = 31) with a measurable clock:
  - all 15 with x within ±0.3 of 3.85 pass (15/15, Wilson lower bound 0.796);
  - 9 of the other 32 pass;
  - every one of the 23 runs whose CE drift over grokking lies in [−5%, +8%] passes.
- **Landing precision.** In T23 the recipe landed x at 3.68–4.10.
- **Expected outcome.** If x lands within ±0.3 of the target, most runs should pass. The depth-3 shares sit closest to the 0.75
  line (77–98%).
- **Power.** The 0.80 bound needs a true pass rate of about 0.93 or more to be reached with good probability. The probability of
  ≥ 37/40 is:

  | true pass rate | P(≥ 37/40) |
  |---|---|
  | 0.85 | 0.13 |
  | 0.90 | 0.42 |
  | 0.93 | 0.69 |
  | 0.95 | 0.86 |

  A miss would be reported as it is, with the point estimate.

## 7. Secondary endpoints (reported in full whatever the primary result)

1. **Recipe vs one fixed rate, paired on the same seed (C2, per depth).**
   - The fixed arm is the full pilot at r₀.
   - Tests: exact McNemar test on primary passes; sign test on |CE drift over grokking|, fixed vs recipe.
2. **Other modulus (C3), width 512 (C4), optimizer ablation (C5).**
   - For C3, C4 and C5: k/n passes with Wilson intervals.
   - For C5: exact McNemar test against the C2 depth-3 recipe runs on the same seeds.
3. **The clock.**
   - Landing: x = r·K₁·t₁₀ of the main run against x\*, and the fraction within ±0.25.
   - CE drift over grokking and its residual from the registered curve −17.1% + 799%·e^(−x).
4. **Protocol-v1 fields on W** (`codex/modarith-atlas` protocol, adopted 2026-09-27), with W in place of the protocol's
   outcome-aware window:
   - joint_PAG10_F02, strict and from the persistent fit;
   - P10/P03 over the transition [t₁₀, first 0.9];
   - representation_first.
5. **Canonical AGOP** (the class-centered AGOP of the protocol, q = 4).
   - Measured for the last and first hidden layers against 8 fixed random frames.
   - A layer counts as aligned when:
     - its purity rises across W by more than the random median does, and
     - it ends W above all 8 random frames.
   - The median gain over random is also given.

These are descriptive. No correction for multiplicity is applied, and none of them changes the primary verdict.

## 8. Analysis procedure

1. **Collect.** `python3 analyze.py collect` fetches every `claude/*` branch of `akashkumar-d/compute`. It takes each chain's
   `results/<chain>/` directory, keeping one copy per chain:
   - a finished copy is preferred over a FAILED record;
   - among finished copies, the owner session's copy, else the earliest finished.
2. **Report.** `python3 analyze.py report` writes `ANALYSIS.md` and `analysis.json`.
3. **When the campaign closes.** It closes when all 164 chains are finished, or 72 hours after the first chain starts, whichever
   comes first.
4. **Chains that crash or are lost.**
   - A chain lost with its session is rerun, either by the owner when it restarts or by another session that takes it over.
   - A chain whose run raises an error three times on one machine is recorded as `FAILED.json`.
   - A session on which two different chains raise errors stops taking chains, and the other sessions take over its chains.
5. **Unfinished chains.** For the primary claim, any chain still unfinished at close counts as a failure. The report also shows
   k over the finished chains.
6. **Machine differences.** Runs on different machines can differ from about step 10 onward: t₁₀ varies by ±2–4% (development
   finding). Every chain records its host, numpy version and code hashes. Nothing is rerun to change an outcome.

## 9. Deviations

Any change after registration is appended to this section with its UTC time and reason. Such changes include code, the manifest,
a threshold, or a stage added or dropped. Each change is also logged in `log/A.md` and in the lab hub.

- **Chains that already started keep their registered code.** A code fix that changes results applies only to chains not yet
  started, and results from before and after the fix are reported separately.
- **Analysis bugs.** A fix that does not change a definition is allowed. It is logged with the old and new numbers. The wording
  in sections 4, 5 and 7 takes precedence over the code.
- **Exploratory analyses.** Anything beyond sections 5 and 7 is labelled exploratory.

1. **2026-09-29 15:45 UTC, analysis code only (no definition changed).**
   - `analyze.py collect` called `Git.remote_branches()` and `Git.fetch()`. `worker.py` 1.1 had renamed these to
     `remote_heads()` and `fetch_changed()` before registration, so `collect` crashed at its first use (156 of 164 chains finished).
   - The fix calls the new names. No endpoint, threshold or window changed.
   - `analyze.py` sha256: 0afaa3330b2f6ea4ee95b1d420b838b74726453c86b0b4270c645e4770e82b14 (registered) ->
     a6931455698e3ca92aff2322ecb927982daaab2c069e59b8c7fa4f98676027cd (used for the analysis).
   - `HASHES.json` keeps the registered hash until the campaign closes, so a worker session restarted from the default branch
     still passes its selftest.
   - The campaign closed on 2026-09-29 at 18:09 UTC (164/164 chains). `HASHES.json` now lists the fixed hash and keeps the
     registered one under `deviations`.

## 10. Registered files

`HASHES.json` lists the SHA-256 of every file that defines the campaign:

- `manifest.json`;
- `make_manifest.py`;
- `natplat_campaign.py`;
- `recipe.py`;
- `worker.py`;
- `endpoints.py`;
- `analyze.py`;
- `tests/test_runner.py`.

`worker.py selftest` checks them in each session before any run.

The hashes of `HASHES.json` and of this file are recorded in `log/A.md` and in the lab hub's LOG at registration.

## 11. Compute

- **Sessions.** Eight Claude Code cloud sessions (S1–S4 on one account, S5–S8 on the other), 4 cores each.
- **Cost.** About 342 core-hours, about 11 hours of wall time.
- **Data flow.**
  - Each session pushes its finished chains and a status file to its own branch.
  - Sessions skip chains finished or running elsewhere.
  - An idle session takes over unstarted chains from sessions that stopped or are far behind.
  - Otherwise an idle session stands by until every chain is finished.
- **Hardware.** No GPU. Nothing runs on Delta.
