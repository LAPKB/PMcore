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

## Deferred post-release work

There is no active implementation slice. The following work is deferred until
after release:

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

- shared-random-stream studies and alternative MCMC kernels;
- Hamiltonian Monte Carlo;
- automatic differentiation and shared sensitivity infrastructure;
- FO, FOCE, and FOCE-I;
- broader dense residual covariance models;
- generic SDE estimation after the explicit particle-session boundary can
  support it without moving likelihood ownership out of PMcore.

New work should default to post-release unless a focused regression demonstrates
incorrect behavior, silent fallback, or misleading output inside the supported
matrix.
