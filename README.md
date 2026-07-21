# PMcore

[![Build](https://github.com/LAPKB/PMcore/actions/workflows/rust.yml/badge.svg)](https://github.com/LAPKB/PMcore/actions/workflows/rust.yml)
[![Documentation](https://github.com/LAPKB/PMcore/actions/workflows/docs.yml/badge.svg)](https://lapkb.github.io/PMcore/pmcore/)
[![Security Audit](https://github.com/LAPKB/PMcore/actions/workflows/security_audit.yml/badge.svg)](https://github.com/LAPKB/PMcore/actions/workflows/security_audit.yml)
[![crates.io](https://img.shields.io/crates/v/pmcore.svg)](https://crates.io/crates/pmcore)

PMcore provides population pharmacokinetic estimation algorithms and result
handling for models executed by `pharmsol`.

## Algorithms

### Nonparametric

- Nonparametric adaptive grid (NPAG)
- Nonparametric optimal design (NPOD)
- Nonparametric maximum a posteriori estimation (NPMAP)
- Non-collapsing NPAG (NCNPAG)

### Parametric SAEM

PMcore supports deterministic analytical and ODE models with:

- identity, log, logit, and probit parameter scales;
- estimated or fixed population parameters;
- inter-individual and inter-occasion variability;
- subject-static continuous and categorical covariate effects;
- parameters and covariate effects with or without IIV;
- explicit fixed, free, and structural-zero covariance entries;
- additive, proportional, combined, correlated-combined, and exponential
  residual models; and
- multiple independently scored outputs.

Eta and kappa are additive in transformed parameter space. Model execution uses
natural parameter values. Covariance declarations use transformed-space
variances and covariances; undeclared covariances are structural zeros.

The default finite schedule ends with `MaxCycles`. `Converged` is available only
through an explicit operational policy whose information, movement, rank,
precision, stationarity, and covariance-stability checks all pass. See
[SAEM convergence](docs/saem-convergence.md).

`FitResult::objf()` and cycle objectives are conditional N2LL values. An
independent opt-in calculation provides population marginal likelihood by
integrating eta and actual-occasion kappa. AIC and BIC are available only when
that marginal N2LL is available; they never fall back to the conditional
objective.

Observed-information covariance and standard errors require an unmodified
strict positive-definite information matrix. They are unavailable when
estimated structural effects omit IIV because structural observation
sensitivities are not implemented. PMcore reports unavailable diagnostics
instead of repairing matrices or fabricating partial uncertainty.

See the [SAEM support matrix](docs/saem-support.md),
[IIV parameterization guide](iiv.md), and
[NONMEM comparison](docs/nonmem-comparison.md) for detailed behavior and syntax.

### Stochastic differential equations

PMcore exposes observation-conditioned particle filtering and bounded diffusion
optimization. Generic SDE fitting through `EstimationProblem` is unsupported;
use `SdeParticleFilter` for filtering.

## Examples

The `examples` directory contains maintained model-fitting and dose-optimization
programs. For example:

```sh
cargo run --example bimodal_ke --release
cargo run --example bimodal_ke_saem --release
```

The SAEM example summarizes the deliberately bimodal data with one Gaussian
random-effects population; it does not reproduce a nonparametric mixture.

## Documentation

API documentation is published at <https://lapkb.github.io/PMcore/>.
