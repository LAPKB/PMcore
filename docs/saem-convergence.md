# SAEM convergence and information diagnostics

PMcore distinguishes finite schedule completion, operational stopping,
information diagnostics, and statistical uncertainty. These are separate
results with separate assumptions.

## Estimator policies

`SaemEstimatorPolicy::TerminalIterate` is the default. A finite schedule that
does not satisfy an enabled operational policy returns `MaxCycles`.

`SaemEstimatorPolicy::AveragedIterates { alpha }` requires `0.5 < alpha < 1`.
During smoothing it uses gain `s^-alpha` and installs the unweighted average of
completed smoothing M-step iterates. Population values are averaged in phi
space, covariance matrices as accepted raw iterates, residual components on
their reported scales, and correlated-combined rho on its raw correlation
scale. Eta is rebased after installation.

An `OperationalConvergenceConfig` is opt-in. It evaluates immutable averaged
candidates at scheduled checkpoints and may stop with `Converged` only when all
configured checks are eligible and satisfied. A failed or ineligible final
check returns `MaxCycles`. This policy does not prove model correctness or
mathematical convergence.

## Free-coordinate order

Information and simulation-variance matrices use this deterministic order:

1. estimated population parameters in declaration order, in phi space;
2. estimated structural lower-triangle Omega entries;
3. estimated structural lower-triangle Omega_IOV entries;
4. estimated residual components by output index: additive, proportional, then
   within-observation correlation.

Fixed values and structural zeros are excluded. Covariance coordinates are raw
covariances. Residual SDs and rho use their reported raw coordinates.

## Observed information

After burn-in, PMcore forms one complete-data replicate per chain by aggregating
all subjects and occasions. Score and Hessian terms are evaluated at the same
pre-M-step parameters and sampled cycle-end latent values used by the SA update.
The current SA gain updates:

```text
Delta <- E[complete score]
C     <- E[complete Hessian]
G     <- E[complete Hessian + score score']
H     = G - Delta Delta'
Iobs  = -H
```

The implementation averages `score score'` across complete chain replicates; it
does not use the outer product of the mean score. Burn-in gain zero leaves the
recursion unchanged.

Derivatives are analytic for Gaussian eta/kappa priors, free raw
Omega/Omega_IOV entries, and every supported residual family. Missing and
non-observation events contribute nothing. Unsupported censoring, invalid
dimensions, non-finite terms, likelihood-floor boundaries, or covariance
failures make information unavailable.

Finite symmetry is accepted only within `64 * f64::EPSILON`. Accepted roundoff
is pairwise averaged before strict Cholesky factorization. PMcore does not add
jitter, ridge, clipping, projection, eigenvalue repair, SVD, or a pseudoinverse.
An indefinite information matrix is retained and labeled rather than repaired.

Population covariance and standard errors are produced only from an unchanged
strict positive-definite observed-information matrix. Identity, log, logit, and
probit coordinates use their exact delta-method transformations. These values
remain unavailable for estimated structural no-IIV effects.

## Frozen-kernel simulation variance

`MarkovSimulationVarianceConfig` explicitly sets the diagnostic seed, chain
count, warmup, retained draws, batch size, lugsail parameters, and trace-memory
limit. No budget is inferred.

After installing the averaged estimate, PMcore starts independent diagnostic
chains from `Normal(0, Omega)` and `Normal(0, Omega_IOV)` draws. Population,
covariance, residual, proposal-scale, and compound-kernel settings remain
fixed. Diagnostic streams do not consume the fit RNG. Each transition runs eta
block attempts, eta component sweeps, and occasion-kappa sweeps without
adaptation or M-steps.

Before allocation or model execution, checked arithmetic calculates a
conservative upper bound for trace storage and workspaces. Exceeding the limit
returns `TraceByteCapExceeded`; arithmetic overflow returns
`TraceMemoryAccountingOverflow`. Raw traces are temporary and are not persisted.

For `n = a b`, nonoverlapping multivariate batch means are

```text
BM_b = b/(a-1) sum_j (mean_j - overall)(mean_j - overall)'
```

and the retained lugsail long-run variance is

```text
Lambda_c = (BM_b - c BM_(b/r)) / (1-c).
```

Chains are never concatenated. PMcore retains both the diagnostic-chain mean LRV
and the fit-operational LRV. With strict observed information,

```text
Xi = Iobs^-1 Lambda_operational Iobs^-T
simulation covariance of the average = Xi / n_avg.
```

Every chain and matrix retains its own typed status. Failed or indefinite chains
remain visible and make the aggregate ineligible. No matrix is projected or
repaired. With no latent dimensions, usable information yields exact zero
simulation-variance matrices without additional model execution.

## Rank and precision checks

PMcore reports rank-normalized split-R-hat, folded split-R-hat, and bulk ESS for
each retained eta and kappa coordinate. Ties use average ranks and Blom scores.
Bulk ESS uses split-chain autocovariances and the initial positive, monotone pair
sequence. Constant traces, odd draw counts, non-finite values, invalid
variances, and insufficient chains receive typed ineligible statuses.

Operational stopping requires every configured information, covariance,
movement, score, eta, and kappa check to be eligible. The supplied rank policy
requires at least four diagnostic chains, maximum R-hat below `1.01`, total bulk
ESS above `400`, and average bulk ESS per split chain of at least `50`.

Relative fixed width is

```text
2 z_(delta/2) * worst_simulation_sd_fraction <= epsilon.
```

Newton displacement is `sqrt(g' Iobs^-1 g)`. Its Monte Carlo SD uses the
diagnostic-mean LRV divided by retained draws. The caller supplies checkpoint
scheduling, confidence, precision, covariance, rejection-window, and
stationarity thresholds.

## Covariance-boundary guardrail

Operational convergence requires an explicit `CovarianceStabilityConfig`. For a
current covariance `Omega` and declared initial covariance `Omega0 = L0 L0'`,
PMcore records

```text
m(Omega; Omega0) = lambda_min(L0^-1 Omega L0^-T).
```

The margin is dimensionless and approaches zero near the positive-definite
boundary. A cycle qualifies only when the margin is at or below the caller's
threshold and the matching covariance update was rejected. Once the declared
consecutive window occurs, operational convergence remains blocked. Recording
this diagnostic does not change the fit trajectory or RNG stream.

## Interpretation

Passing the operational policy means only that the configured numerical and
sampling checks passed for that fit. Stationarity, adequate mixing, the Markov
Poisson equation, and controlled-Markov stochastic-approximation assumptions
remain unverified. Information and simulation-variance matrices are not by
themselves proof of convergence. For consequential use, compare an independent
fit with larger MCMC and schedule budgets.

Population marginal likelihood and AIC/BIC are separate post-fit calculations.
They do not change operational stopping and do not turn a conditional objective
into population evidence.

## References

- Delyon, B., Lavielle, M., and Moulines, E. (1999). Convergence of a stochastic
  approximation version of the EM algorithm. *Annals of Statistics* 27(1),
  94-128.
- Kuhn, E., and Lavielle, M. (2004). Coupling a stochastic approximation version
  of EM with an MCMC procedure. *ESAIM: Probability and Statistics* 8, 115-131.
- Vehtari, A. et al. (2021). Rank-normalization, folding, and localization.
  *Bayesian Analysis* 16(2), 667-718.
- Vats, D., and Flegal, J. M. (2022). Lugsail lag windows for estimating
  time-average covariance matrices. *Biometrika* 109(3), 735-750.
