# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SAEM for deterministic analytical and ODE models with IIV, IOV, and supported
  residual-error models.
- Assay and residual likelihood scoring for estimation objectives.
- Explicit SDE particle filtering and bounded diffusion optimization.
- Deterministic one-compartment analytical/ODE prediction and normalized
  conditional-objective parity coverage.
- Typed parametric numerical-failure termination that returns no partial fit result.
- SAEM lifecycle, cycle diagnostics, guardrail, and truthful termination tracing.
- Owned live parametric fit snapshots, clamped cycle progress, post-cycle
  observers, reason-preserving user/stop-file termination, and stale stop-file cleanup.
- SAEM results retain the original equation, data, and ordered parameter names
  for prediction and follow-up-run APIs.
- Parametric results generate population and conditional predictions on demand.
- Parametric results retain requested SAEM configuration, effective chain count,
  ordered parameter metadata, and covariance masks.
- Parametric results provide structured population, covariance, residual,
  individual, iteration, statistic, and prediction tables with table-owned CSV
  writers, plus a versioned equation-free JSON result and output manifest.
- Population parameter summaries distinguish the primary estimate from optional
  distribution and uncertainty statistics. Natural-scale standard deviations
  and coefficients of variation are present only for free population parameters
  with available strict observed-information uncertainty; unsupported and fixed
  coordinates remain absent.
- Deterministic free-coordinate metadata, analytic complete-data score/Hessian
  recursion, and immutable observed-information diagnostics. Residual derivatives
  exactly follow the active likelihood scale floor, canonical proportional
  coordinates join persisted residual rows, and every long-form information row
  carries its availability status. These remain diagnostic rather than standard
  errors.
- An opt-in `AveragedIterates { alpha }` SAEM policy with compatible smoothing
  gain, direct phi/raw-covariance/raw-SD Cesaro averaging, eta rebasing, canonical
  post-average result recomputation and result metadata. Terminal-iterate
  trajectories remain the default and unchanged.
- An opt-in frozen-kernel Markov simulation-variance and mixing diagnostic with
  explicit chain/draw/memory budgets and independently seeded prior draws at the
  frozen averaged Omega/Omega_IOV. The same retained transitions provide full
  complete-score and eta/kappa traces, per-chain multivariate lugsail batch
  means, diagnostic-mean and fit-operational LRV scales, rank-normalized and
  folded split-Rhat, and bulk ESS. Checked trace allocation fails before model
  execution; invalid coordinates retain typed statuses without hiding valid
  coordinates. A separate explicit operational policy applies Vehtari rank
  thresholds, Gong/Flegal relative fixed width, and caller-supplied PMcore
  stationarity thresholds at deterministic checkpoints. A joint pass yields
  `Converged`; the default and every failed/ineligible finite schedule yield
  `MaxCycles`. Stationarity, mixing, Poisson-equation, and controlled-Markov CLT
  assumptions remain unverified, and no uncertainty claim is made. Derived
  normal-quantile/implied-ESS values now fail validation before fitting when
  unusable; lifecycle warnings distinguish unevaluated, failed, ineligible, and
  passed checks; and operational CSV rows retain explicit status for every
  criterion, trace statistic, LRV, and information-mapped matrix. Accepted
  machine-roundoff asymmetry is canonicalized only after the finite-
  symmetry tolerance succeeds, preventing exact-symmetry factorization from
  misclassifying a positive-definite matrix without jitter or repair.
- Caller-declared covariance-stability diagnostics record a scale-invariant
  generalized SPD margin for Omega/Omega_IOV, emit typed warnings after a
  complete consecutive near-boundary rejection window, and make operational
  convergence fail closed after such a run. Thresholds and windows have no
  PMcore default and do not alter fixed-schedule trajectories.
- Explicit opt-in post-fit population marginal likelihood jointly integrates
  eta and each actual-occasion kappa with normalized Student-t importance
  sampling around retained conditional modes. Exact no-latent evaluation,
  independent subject streams, ESS, zero-weight counts, delta-method N2LL MCSE,
  typed unavailable/nonconverged-mode statuses, immutable result accessors,
  warm-start recomputation semantics, and complete CSV/JSON output are included.
  The compatibility objective and all conditional APIs remain unchanged.
- Pure post-fit AIC and independent-subject BIC derived only from available
  population marginal N2LL, with deterministic free-coordinate counts, exact
  N2LL-MCSE propagation, typed availability, result/summary accessors, and
  status-bearing CSV/statistics output. Criteria never use the conditional
  compatibility objective.
- Parametric result and manifest schema 9 retains typed marginal-likelihood,
  information-criteria, uncertainty, correlated-residual, shrinkage,
  information, and Markov diagnostics. Schema versions 1-8 are intentionally
  rejected; derived rows are recomputed and checked against retained inputs.
- Masked Louis observed information supplies strict unregularized population
  covariance and exact identity/log/logit/probit delta-method standard errors.
  One joint eta/kappa central-difference curvature supplies conditional
  covariance and standard errors plus an opt-in Student-t proposal. Raw Omega
  blocks remain the default proposal. Shrinkage uses unclamped `N-1` sample
  variance for separately named posterior-mean and MAP eta/kappa.
- A scalar within-observation correlated additive/proportional residual family
  implements `Var(Y|f) = a² + 2 rho a b f + b²f²`, independent fixed/free
  controls, log-SD/Fisher-rho optimization, analytic Louis derivatives, IOV
  coexistence, and schema-9 lifecycle support. It does not imply serial,
  cross-time, cross-output, or general block-sigma residual correlation.
- Support for numerically estimating population and covariate effects
  without IIV from the current latent-trace observation likelihood, using the
  ordinary SAEM gain. Observed-information uncertainty remains unsupported.
- Explicit variance- and SD-based diagonal constructors for `Omega` and `Iov`;
  legacy `diagonal` remains variance-based and SD overflow fails closed.

### Changed

- Exponential residual likelihood scoring now uses the same machine-scale floor
  as the canonical residual scale and observed-information derivatives.
- Integrate with pharmsol prediction and simulation APIs.
- Generic SDE fitting is unsupported; use `SdeParticleFilter` for
  observation-conditioned filtering.
- Define the deterministic analytical/ODE SAEM support matrix and fail closed on
  invalid parameter, data, residual, and operational configuration values.

### Fixed

- Update coupled covariate raw first/second moments with one common SAEM gain so
  the Gaussian moment history remains realizable. Apply exploration robustness
  as an objective-checked, mask-preserving under-relaxation of the accepted
  Omega/GEM displacement, with no second smoothing gain. Capped exploration
  floors the solved target before interpolation; uncapped covariate smoothing
  and non-covariate IIV/IOV preserve legacy floor-after-interpolation
  backtracking. Reject estimated initial Omega/Omega_IOV diagonals below their
  configured floors while exempting fixed diagonals.
- Keep operational covariance-rejection criteria unavailable until a complete
  active-cycle window exists; a short healthy prefix no longer satisfies a
  longer configured consecutive window.
- Correct Vehtari/Geyer bulk ESS pair indexing and tau assembly, retain separate
  rank-Rhat/folded-Rhat/ESS statuses, preserve failed LRV chain indices, and
  report checked required versus actually allocated peak trace memory.
- Update IIV covariance from a dedicated eta second-moment statistic after
  population recentering instead of mixing differently stepped phi moments.
- Apply annealing, exploration replacement, and smoothing steps independently to
  estimated combined-residual components while preserving fixed components.
- Assemble default MAP-enabled results for models without eta or kappa dimensions.
- Reconstruct IOV individual-parameter output separately for every subject and
  occasion from the matching eta and kappa source.
- Reject rank-deficient covariance updates with scale-stable factorization while
  retaining finite high-scale positive-definite candidates.
- Preserve sparse residual-output indices and valid fixed-zero combined-error
  components when reconstructing parametric warm starts and when accumulating,
  validating, and installing averaged SAEM estimates.

## [0.26.1](https://github.com/LAPKB/PMcore/compare/v0.26.0...v0.26.1) - 2026-07-20

### Fixed

- Normalize psi to prevent likelihood underflow ([#300](https://github.com/LAPKB/PMcore/pull/300))

### Other

- Update pharmsol to 0.28.2 ([#303](https://github.com/LAPKB/PMcore/pull/303))

## [0.26.0](https://github.com/LAPKB/PMcore/compare/v0.25.2...v0.26.0) - 2026-07-13

### Added

- Update BestDose API ([#296](https://github.com/LAPKB/PMcore/pull/296))
- Add controller to step through a fit manually ([#294](https://github.com/LAPKB/PMcore/pull/294))
- Chain fit methods using result ([#284](https://github.com/LAPKB/PMcore/pull/284))
- Support parametric and non-parametric algorithms ([#276](https://github.com/LAPKB/PMcore/pull/276))

### Fixed

- Fix outputs generation ([#288](https://github.com/LAPKB/PMcore/pull/288))

### Other

- Bump pharmsol to 0.28.1 ([#298](https://github.com/LAPKB/PMcore/pull/298))
- Update documentation ([#297](https://github.com/LAPKB/PMcore/pull/297))
- Bump pharmsol ([#295](https://github.com/LAPKB/PMcore/pull/295))
- Add missing implementations ([#292](https://github.com/LAPKB/PMcore/pull/292))
- Remove unused dependencies ([#291](https://github.com/LAPKB/PMcore/pull/291))
- Update security audit workflow ([#289](https://github.com/LAPKB/PMcore/pull/289))

## [0.25.2](https://github.com/LAPKB/PMcore/compare/v0.25.1...v0.25.2) - 2026-04-20

### Fixed

- Add wrappers for analytical solutions ([#272](https://github.com/LAPKB/PMcore/pull/272))

## [0.25.1](https://github.com/LAPKB/PMcore/compare/v0.25.0...v0.25.1) - 2026-04-13

### Other

- Update rand requirement from 0.9.0 to 0.10.1 ([#270](https://github.com/LAPKB/PMcore/pull/270))

## [0.25.0](https://github.com/LAPKB/PMcore/compare/v0.24.0...v0.25.0) - 2026-04-11

### Added

- Bump pharmsol and update examples ([#269](https://github.com/LAPKB/PMcore/pull/269))
- Use pharmsol 0.25 ([#268](https://github.com/LAPKB/PMcore/pull/268))

### Other

- Update faer requirement from 0.23.1 to 0.24.0 ([#241](https://github.com/LAPKB/PMcore/pull/241))
- Update rand requirement from 0.9.0 to 0.10.0 ([#244](https://github.com/LAPKB/PMcore/pull/244))

## [0.24.0](https://github.com/LAPKB/PMcore/compare/v0.23.0...v0.24.0) - 2026-04-01

### Added

- new bestdose API ([#247](https://github.com/LAPKB/PMcore/pull/247))

### Other

- Update CI workflows ([#258](https://github.com/LAPKB/PMcore/pull/258))
- fix examples to match the data ([#259](https://github.com/LAPKB/PMcore/pull/259))

## [0.23.0](https://github.com/LAPKB/PMcore/compare/v0.22.2...v0.23.0) - 2026-03-24

### Other

- Bump pharmsol to 0.24 ([#251](https://github.com/LAPKB/PMcore/pull/251))

## [0.22.2](https://github.com/LAPKB/PMcore/compare/v0.22.1...v0.22.2) - 2025-12-12

### Other

- Update pharmsol dependency version to 0.22.1 ([#236](https://github.com/LAPKB/PMcore/pull/236))
- Update criterion requirement from 0.7 to 0.8 ([#233](https://github.com/LAPKB/PMcore/pull/233))
- Update pharmsol requirement from =0.21.0 to =0.22.0 ([#232](https://github.com/LAPKB/PMcore/pull/232))

## [0.22.1](https://github.com/LAPKB/PMcore/compare/v0.22.0...v0.22.1) - 2025-11-18

### Added

- methods needed to be able to separate problem from fit in Pmetrics ([#228](https://github.com/LAPKB/PMcore/pull/228))

## [0.22.0](https://github.com/LAPKB/PMcore/compare/v0.21.1...v0.22.0) - 2025-11-17

### Added

- [**breaking**] Update NPResult to contain the posterior and predictions ([#224](https://github.com/LAPKB/PMcore/pull/224))

### Other

- Remove unused dependency (argmin-math) ([#225](https://github.com/LAPKB/PMcore/pull/225))

## [0.21.1](https://github.com/LAPKB/PMcore/compare/v0.21.0...v0.21.1) - 2025-11-12

### Other

- Update field name in NPPredictionrow to cens ([#222](https://github.com/LAPKB/PMcore/pull/222))

## [0.21.0](https://github.com/LAPKB/PMcore/compare/v0.20.0...v0.21.0) - 2025-11-05

### Added

- *(exa)* [**breaking**] exa now requires the path where the template is going to be deployed ([#219](https://github.com/LAPKB/PMcore/pull/219))
- Implement BestDose ([#163](https://github.com/LAPKB/PMcore/pull/163))

### Fixed

- Remove algorithms from .gitignore ([#220](https://github.com/LAPKB/PMcore/pull/220))

### Other

- Refactor algorithm trait ([#196](https://github.com/LAPKB/PMcore/pull/196))
- Algorithm is Send+static

## [0.20.0](https://github.com/LAPKB/PMcore/compare/v0.19.1...v0.20.0) - 2025-10-24

### Other

- Update field to censoring and bump pharmsol
- Add censoring to output data
- Breaking Changes in Pharmsol
- Bolus are not given automatically anymore

## [0.19.1](https://github.com/LAPKB/PMcore/compare/v0.19.0...v0.19.1) - 2025-10-23

### Other

- Bump pharmsol ([#211](https://github.com/LAPKB/PMcore/pull/211))

## [0.19.0](https://github.com/LAPKB/PMcore/compare/v0.18.1...v0.19.0) - 2025-10-22

### Added

- New version of pharmsol ([#208](https://github.com/LAPKB/PMcore/pull/208))
- Implement serialize for NPResult ([#202](https://github.com/LAPKB/PMcore/pull/202))

### Other

- Remove writing of the op.csv ([#205](https://github.com/LAPKB/PMcore/pull/205))
- Add more tests ([#199](https://github.com/LAPKB/PMcore/pull/199))
- More informative error for Cholesky decomposition ([#198](https://github.com/LAPKB/PMcore/pull/198))

## [0.18.1](https://github.com/LAPKB/PMcore/compare/v0.18.0...v0.18.1) - 2025-10-09

### Added

- Expose methods on Theta ([#200](https://github.com/LAPKB/PMcore/pull/200))

## [0.18.0](https://github.com/LAPKB/PMcore/compare/v0.17.0...v0.18.0) - 2025-09-30

### Added

- all optimization routines moved to pharmsol, it makes more sense for them to be there since there are somoe optimization routines that belong to the model and not to a population algorithm

### Other

- Update argmin requirement from 0.10.0 to 0.11.0 ([#192](https://github.com/LAPKB/PMcore/pull/192))

## [0.17.0](https://github.com/LAPKB/PMcore/compare/v0.16.0...v0.17.0) - 2025-09-29

### Added

- Refactor outputs ([#178](https://github.com/LAPKB/PMcore/pull/178))

### Fixed

- Duplicate predictions ([#187](https://github.com/LAPKB/PMcore/pull/187))

### Other

- Greco model ([#190](https://github.com/LAPKB/PMcore/pull/190))
- Update faer requirement from 0.22.4 to 0.23.1 ([#186](https://github.com/LAPKB/PMcore/pull/186))

## [0.15.1](https://github.com/LAPKB/PMcore/compare/v0.15.0...v0.16.0) - 2025-09-07

### Added

- Support pharmsol fixed error factors ([#176](https://github.com/LAPKB/PMcore/pull/176))
- Update calculation of predictions ([#159](https://github.com/LAPKB/PMcore/pull/159))

### Other

- Drusano Greco model ([#182](https://github.com/LAPKB/PMcore/pull/182))
- Update pharmsol requirement from =0.16.0 to =0.17.0 ([#183](https://github.com/LAPKB/PMcore/pull/183))
- Update pharmsol requirement from =0.15.0 to =0.16.0 ([#177](https://github.com/LAPKB/PMcore/pull/177))
- Improve test coverage ([#172](https://github.com/LAPKB/PMcore/pull/172))
- Bump actions/checkout from 4 to 5 ([#175](https://github.com/LAPKB/PMcore/pull/175))
- Update pharmsol requirement from =0.14.0 to =0.15.0 ([#173](https://github.com/LAPKB/PMcore/pull/173))
- Improve benchmark coverage ([#174](https://github.com/LAPKB/PMcore/pull/174))
- Change log-level of output folder destination ([#158](https://github.com/LAPKB/PMcore/pull/158))
- Update criterion requirement from 0.6 to 0.7 ([#169](https://github.com/LAPKB/PMcore/pull/169))

## [0.15.0](https://github.com/LAPKB/PMcore/compare/v0.14.0...v0.15.0) - 2025-07-23

### Added

- v0.15.0

### Other

- support for pharmsol 0.14.0

## [0.14.0](https://github.com/LAPKB/PMcore/compare/v0.13.1...v0.14.0) - 2025-07-14

### Added

- Update output files API ([#149](https://github.com/LAPKB/PMcore/pull/149))

## [0.13.1](https://github.com/LAPKB/PMcore/compare/v0.13.0...v0.13.1) - 2025-07-09

### Added

- support for pharmsol 0.13.1 ([#153](https://github.com/LAPKB/PMcore/pull/153))

## [0.13.0](https://github.com/LAPKB/PMcore/compare/v0.12.1...v0.13.0) - 2025-06-25

### Added

- More informative status for algorithm and stop reason ([#138](https://github.com/LAPKB/PMcore/pull/138))

### Fixed

- Add gamma/lambda for each output equation to cycle.csv ([#147](https://github.com/LAPKB/PMcore/pull/147))

### Other

- Update Cargo.toml ([#150](https://github.com/LAPKB/PMcore/pull/150))
- Add validation of parameters ([#136](https://github.com/LAPKB/PMcore/pull/136))

## [0.12.1](https://github.com/LAPKB/PMcore/compare/v0.12.0...v0.12.1) - 2025-06-19

### Other

- Update Cargo.toml ([#145](https://github.com/LAPKB/PMcore/pull/145))
- Use pharmsol 0.11.0

## [0.12.0](https://github.com/LAPKB/PMcore/compare/v0.11.0...v0.12.0) - 2025-06-11

### Added

- Deprecate fixed but unknown ([#118](https://github.com/LAPKB/PMcore/pull/118))

### Other

- Error model ([#139](https://github.com/LAPKB/PMcore/pull/139))

## [0.10.0](https://github.com/LAPKB/PMcore/compare/v0.9.0...v0.10.0) - 2025-03-28

### Added

- API changes ([#112](https://github.com/LAPKB/PMcore/pull/112))

### Other

- Update README.md ([#114](https://github.com/LAPKB/PMcore/pull/114))

## [0.9.0](https://github.com/LAPKB/PMcore/compare/v0.8.2...v0.9.0) - 2025-03-25

### Added

- Use `faer` for linear algebra ([#108](https://github.com/LAPKB/PMcore/pull/108))

## [0.8.2](https://github.com/LAPKB/PMcore/compare/v0.8.1...v0.8.2) - 2025-03-17

### Fixed

- Error in logic for output folder ([#109](https://github.com/LAPKB/PMcore/pull/109))

## [0.8.1](https://github.com/LAPKB/PMcore/compare/v0.8.0...v0.8.1) - 2025-03-12

### Fixed

- Parameters from Vec<Parameter> ([#106](https://github.com/LAPKB/PMcore/pull/106))

## [0.8.0](https://github.com/LAPKB/PMcore/compare/v0.7.6...v0.8.0) - 2025-03-12

### Added

- Refactor settings API ([#101](https://github.com/LAPKB/PMcore/pull/101))

### Other

- Update QR-decomposition ([#97](https://github.com/LAPKB/PMcore/pull/97))
