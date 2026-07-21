# SAEM support

PMcore validates the model, data, parameter, covariance, residual, and runtime
configuration before fitting. Unsupported combinations fail with an error
instead of selecting a fallback.

## Models and data

| Area | Supported behavior |
| --- | --- |
| Equations | Deterministic analytical and ODE equations with complete metadata |
| Subjects | One or more subjects and at least one measured observation |
| Outputs | Explicit metadata names; numeric `N` requires a declared `outeq_N` |
| Missing values | Retained in the event stream and omitted from scoring |
| Censoring | Not supported for parametric estimation |
| Covariates | Finite subject-static continuous and categorical values |
| Assay metadata | `ErrorPoly` C0-C3 values are transported but do not select parametric scoring |

Every measured output requires an explicit `ParametricErrorModel`. Population
covariate effects are linear in transformed parameter space. Covariate values
must be present, finite, and constant within each subject.

PMcore executes each ODE with its configured solver and tolerances. Solver
choice is a scientific model input; PMcore does not replace it during fitting.
Stiff models generally require an implicit solver and tolerances selected for
the model's scale. Completion of a finite SAEM schedule does not establish ODE
accuracy.

## Parameters and variability

| Area | Supported behavior |
| --- | --- |
| Scales | Identity, Log, Logit, Probit |
| Population values | Independently fixed or estimated, with or without IIV |
| IIV | Named parameter subsets or zero-dimensional |
| IOV | Named parameter subsets, independently of IIV |
| Covariance | Fixed/free entries, structural zeros, strict positive definiteness |

Eta and kappa are additive in transformed parameter space. Model execution uses
natural parameter values. `Parameter::with_initial` is the natural-scale value
at zero eta, kappa, and covariate offsets.

`Omega::diagonal_variances` and `Iov::diagonal_variances` accept variances.
`diagonal_standard_deviations` accepts finite positive SDs and checks overflow
before squaring. Legacy `diagonal` remains variance-based. Undeclared
covariances are structural zeros.

Covariance updates preserve fixed entries and structural zeros. Invalid,
non-finite, non-symmetric, or non-positive-definite matrices are rejected; no
jitter, ridge, clipping, projection, eigendecomposition repair, or pseudoinverse
is used.

Estimated no-IIV population and covariate coordinates use the observation
likelihood. Population observed-information covariance and standard errors are
reported as unsupported while those coordinates are estimated.

## Residual models

| Family | Parameters |
| --- | --- |
| Constant | fixed or estimated SD |
| Proportional | fixed or estimated coefficient |
| Combined | independently fixed or estimated additive and proportional components |
| Correlated combined | additive SD, proportional coefficient, and within-observation correlation |
| Exponential | fixed or estimated log-scale SD |

For correlated combined error,

```text
Var(Y | f) = a^2 + 2 rho a b f + b^2 f^2
```

with finite `a,b > 0` and `-1 < rho < 1`. Correlation is scalar and applies only
to the additive and proportional components of one observation. Serial,
cross-time, cross-output, dense, and general block residual covariance are not
supported. Multiple outputs use independent named residual declarations.

## Schedule and MCMC

A valid schedule has `k1 + k2 > 0`, `burn_in <= k1`, and no integer overflow.
Burn-in performs MCMC without parameter updates. Exploration uses gain one.
Smoothing uses a decreasing gain. The default estimator returns the terminal
iterate. `AveragedIterates { alpha }` requires `k2 > 0` and `0.5 < alpha < 1`.

MCMC chain counts, iteration counts, adaptation intervals, and proposal scales
must be positive. `eta_block_iterations = 0` disables eta block proposals. Raw
Omega blocks are the default block scale. Conditional-curvature scaling is
opt-in and fails with a typed status when strict curvature is unavailable.

Covariate raw first and second moments always use the same SA gain. PMcore forms
a centered covariance target before applying masks and the constrained local
GEM update. Exploration may under-relax the accepted displacement; smoothing
applies no second covariance gain.

## Objectives and uncertainty

`FitResult::objf()`, cycle records, and compatibility summaries contain
conditional N2LL. They are diagnostics and never select a fit or substitute for
population evidence.

Population marginal likelihood is an explicit post-fit calculation. It jointly
integrates eta and actual-occasion kappa with normalized Student-t importance
sampling, or evaluates the observation likelihood exactly when there are no
latent dimensions. Results include ESS, zero-weight counts, and delta-method
N2LL Monte Carlo error. AIC and BIC are derived only from available marginal
N2LL and retain that MC error.

Observed information uses analytic complete-data derivatives in a deterministic
free-coordinate order. Population covariance and standard errors require an
unmodified strict-Cholesky inverse. Conditional eta/kappa uncertainty uses one
joint central-difference curvature in `[eta, kappa_1, ..., kappa_K]` order.
Unavailable curvature or information remains unavailable without fallback.

Shrinkage is reported separately for posterior-mean and MAP eta/kappa sources.
It uses `100 * (1 - sample_variance / population_variance)` with `N-1` sample
variance and is not clamped.

## Results and lifecycle

Results retain the equation, data, ordered parameter metadata, covariance masks,
requested configuration, effective chain count, cycle diagnostics, predictions,
conditional modes, uncertainty statuses, and optional marginal likelihood.
Schema 9 is current; older schemas are rejected.

The controller supports cycle stepping, post-cycle observers, owned snapshots,
user abort, stop-file termination, and truthful terminal reasons. A stale
current-directory `stop` file is removed before a new run.

The default finite schedule reports `MaxCycles`. An opt-in operational policy
may report `Converged` only when all configured information, movement,
rank-normalized R-hat, bulk ESS, relative fixed-width, stationarity, and
covariance-stability checks pass. See [SAEM convergence](saem-convergence.md).

## Unsupported

- Generic SDE fitting through `EstimationProblem`; use `SdeParticleFilter` or
  bounded diffusion optimization.
- Parametric BLOQ or ALOQ censoring.
- Time-varying population covariate effects.
- Arbitrary nonlinear parameter constraints.
- Serial or multivariate residual covariance.
- Observed-information covariance for estimated structural no-IIV effects.
- Automatic theorem-level convergence claims.
- FO, FOCE, and FOCE-I.
