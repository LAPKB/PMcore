# SAEM current status and future work

## Current implementation

PMcore provides a production SAEM path for deterministic analytical and ODE
models. The implementation includes:

- transformed-space population parameters with identity, log, logit, and probit
  scales;
- IIV and IOV with named fixed/free covariance masks and structural zeros;
- subject-static continuous and categorical covariate effects;
- estimated population and covariate effects with or without IIV;
- additive, proportional, combined, correlated-combined, and exponential
  residual models;
- persistent eta and kappa MCMC with component and opt-in block proposals;
- burn-in, exploration, and decreasing-gain smoothing phases;
- terminal-iterate and opt-in averaged estimators;
- strict observed-information and conditional-curvature diagnostics;
- eta and kappa posterior-mean/MAP shrinkage;
- population and conditional predictions;
- post-fit population marginal likelihood, AIC, and BIC;
- cycle-by-cycle controllers, observers, snapshots, and typed termination;
- schema-9 persistence, structured outputs, and warm starts; and
- explicit particle filtering and bounded diffusion optimization for SDE use.

Covariate raw first and second moments use one common SA gain. PMcore forms the
centered covariance target before applying masks, local GEM constraints, strict
positive-definiteness checks, and any exploration-only displacement cap.
Smoothing does not apply a second covariance gain.

The default finite SAEM schedule reports `MaxCycles`. `Converged` is available
only through an explicit operational policy. Conditional N2LL remains a
diagnostic; it never substitutes for population marginal likelihood.

The support matrix and failure semantics are maintained in
[`docs/saem-support.md`](../docs/saem-support.md). Convergence and information
semantics are maintained in
[`docs/saem-convergence.md`](../docs/saem-convergence.md).

## Must Have — completed release blockers

M7 and M8 are implemented and validated in the current branch.

### M7 — Canonical parameter ordering — Complete

PMcore now validates parameter declarations by name and canonicalizes them to
model metadata order before constructing covariates, Omega, IOV, scoring state,
persistence metadata, or results.

- Canonicalize every parameter-aligned structure to
  `model.parameter_names()` before model execution, scoring, diagnostics,
  persistence, and result construction.
- Preserve name-based Omega, IOV, covariate, fixed/free, warm-start, and
  persistence semantics through the reorder.
- Continue to reject duplicate, unknown, and missing declarations explicitly.
- Add analytical and ODE regressions proving that out-of-order declarations
  produce the same predictions, objectives, estimates, and labels as canonical
  declarations.
- Do not retain a positional fallback or merely document the unsafe ordering
  requirement.

Completion evidence includes ODE metadata/Omega/IOV resolution and exact
analytical fit, objective, diagnostic, and prediction parity for reordered
declarations.

### M8 — SAEMix four-kernel compatibility — Complete

PMcore retains its established component/full-Omega-block policy and now offers
an explicit SAEMix-compatible IIV policy implementing kernels 1 through 4. The
compatibility policy fails closed for IOV; PMcore's established eta/kappa policy
continues to support IOV.

- Add explicit iteration counts for the prior-independence, componentwise,
  rotating-subset, and early MAP-informed kernels without overloading the
  existing post-fit MAP controls.
- Implement the kernels in SAEMix order, including the rotating subset-size
  schedule, the early-cycle MAP proposal window, Metropolis-Hastings proposal
  corrections, and SAEMix-compatible proposal-scale adaptation.
- Retain the current PMcore kernel policy as an explicit supported mode; exact
  SAEMix behavior must be selected deliberately rather than introduced as a
  silent default change.
- Record proposals, acceptance, non-finite rejection, and adapted scales
  separately for each kernel in cycle diagnostics and controller snapshots.
- Add deterministic kernel-level tests and cross-engine tests on equivalent
  parameterizations. Cross-engine acceptance must compare estimates and
  distributions over multiple seeds, not identical RNG trajectories.
- Keep post-fit `compute_map` behavior distinct from the in-E-step MAP-informed
  kernel.

Completion evidence includes deterministic kernel tests and a bounded
three-seed PMcore/SAEMix theophylline panel with equivalent `ka/V/ke`, diagonal
Omega, residual, schedule, and `c(2,2,2,2)` settings. Mean PMcore-versus-SAEMix
differences were -3.40% for ka, -1.13% for V, +2.68% for ke, +0.31% for sigma,
and -5.70% for the estimable ke variance; run products remain outside the
repository.

The release validation commands pass after both slices:

- `cargo fmt --check`
- `cargo check`
- `cargo test saem --lib`
- `cargo test parametric --lib`
- `cargo test likelihood --lib`

## Deferred post-release work

The following work remains deferred post-release:

### Reference-model coverage

- Add one maintained large-model regression that exercises the public model,
  covariate, residual, persistence, and result APIs without creating a separate
  validation framework.
- Expand replicated analytical and ODE coverage only when each fixture protects
  a concrete supported behavior.
- Add broader cross-engine comparisons only as bounded development work; keep
  external run products outside the product repository.

### Statistical maturity

- Evaluate convergence and coverage over larger replicated datasets.
- Improve marginal-likelihood proposal diagnostics and ambiguity handling.
- Extend uncertainty reporting where structural observation sensitivities are
  available.
- Add shrinkage and information summaries for new supported coordinate types.

### Lifecycle maturity

- Bring nonparametric persistence and lifecycle APIs to the same level as the
  parametric controller.
- Review result-schema evolution before adding new persisted diagnostics.
- Keep package examples small, self-contained, and runnable.

## Optional research

These are not release commitments:

- shared-random-stream studies and alternative MCMC kernels beyond the required
  SAEMix-compatible four-kernel policy;
- Hamiltonian Monte Carlo;
- automatic differentiation and shared sensitivity infrastructure;
- FO, FOCE, and FOCE-I;
- broader dense residual covariance models;
- generic SDE estimation after the explicit particle-session boundary can
  support it without moving likelihood ownership out of PMcore.

New work should default to post-release unless a focused regression demonstrates
incorrect behavior, silent fallback, or misleading output inside the supported
matrix.
