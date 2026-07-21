# NONMEM and PMcore model declarations

This guide maps common NONMEM declarations to PMcore's parametric API. It is a
syntax and parameterization comparison, not a control-stream converter.

## Concepts

| NONMEM | PMcore |
| --- | --- |
| `$THETA` | `Parameter` declarations and `with_initial` |
| `$OMEGA` | `Omega` for IIV and `Iov` for IOV |
| `ETA(n)` | named eta associated with a parameter |
| occasion-specific ETA terms | named kappa generated from `Iov` |
| `$SIGMA` and `$ERROR` | explicit `ParametricErrorModel` per output |
| `$PK` covariate equations | `CovariateEffect` in transformed phi space |
| `$ESTIMATION` | `SaemConfig` and `fit_with` |

PMcore uses names rather than numeric ETA, kappa, output, and covariance
positions. Model macros supply equation metadata used to validate those names.

## Population values and IIV

NONMEM log-normal clearance:

```text
$THETA
(0, 5)

$OMEGA
0.09

$PK
CL = THETA(1) * EXP(ETA(1))
```

PMcore:

```rust
.parameter(Parameter::log("cl").with_initial(5.0))
.omega(Omega::diagonal_variances([("cl", 0.09)]))
```

`Parameter::log` means eta is additive on the log scale. The model receives the
natural value `5 * exp(eta)`. `Omega` values are variances, not standard
deviations.

For additive IIV, use `Parameter::real`. Bounded logit and probit declarations
store their bounds and transformations directly instead of requiring a manual
inverse transformation in the model equation.

## Correlated IIV

NONMEM:

```text
$OMEGA BLOCK(2)
0.09
0.01 0.04
```

PMcore:

```rust
.omega(
    Omega::diagonal_variances([
        ("cl", 0.09),
        ("v", 0.04),
    ])
    .covariance("cl", "v", 0.01)
)
```

Undeclared PMcore covariances are structural zeros. Use `fixed_variance` and
`fixed_covariance` for fixed entries. PMcore rejects a declaration that is not
finite, symmetric, and strictly positive definite.

## Inter-occasion variability

A NONMEM model often selects a distinct ETA by occasion:

```text
$OMEGA
0.09 ; IIV variance for CL
0.04 ; occasion 1 variance
0.04 ; occasion 2 variance

$PK
KAPPA = 0
IF (OCC.EQ.1) KAPPA = ETA(2)
IF (OCC.EQ.2) KAPPA = ETA(3)
CL = THETA(1) * EXP(ETA(1) + KAPPA)
```

PMcore declares one kappa distribution and creates one draw for every actual
occasion:

```rust
.parameter(Parameter::log("cl").with_initial(5.0))
.omega(Omega::diagonal_variances([("cl", 0.09)]))
.iov(Iov::diagonal_variances([("cl", 0.04)]))
```

The individual occasion value is

```text
CL_i,k = TVCL * exp(eta_i + kappa_i,k).
```

In builder-created data, `SubjectBuilderExt::reset()` starts the next occasion:

```rust
let subject = Subject::builder("1")
    .bolus(0.0, 100.0, "iv")
    .observation(1.0, 2.1, "cp")
    .reset()
    .bolus(0.0, 100.0, "iv")
    .observation(1.0, 1.9, "cp")
    .build();
```

Imported data retains its `Subject -> Occasion -> Event` hierarchy. Kappa is
indexed by subject, actual occasion, and named IOV effect. `Iov` supports the
same fixed/free entries, structural zeros, and variance/SD constructors as
`Omega`.

## Parameters without IIV

NONMEM omits ETA from the parameter expression:

```text
$PK
BASE = THETA(1)
```

PMcore makes the choice explicit:

```rust
.parameter(
    Parameter::real("baseline")
        .with_initial(1.0)
        .without_random_effect()
)
```

The population value may still be estimated. PMcore estimates structural no-IIV
population and covariate effects from the observation likelihood. Their
observed-information covariance is currently unavailable.

## Covariates

A PMcore continuous effect is linear in transformed phi space. For a log
parameter,

```rust
.covariate_effect(
    CovariateEffect::continuous("cl", "wt", 70.0)
        .with_initial(0.01)
)
```

defines

```text
log(CL_i) = log(TVCL) + 0.01 * (WT_i - 70) + eta_i.
```

The comparable NONMEM expression is:

```text
$PK
CL = THETA(1) * EXP(THETA(2) * (WT - 70) + ETA(1))
```

Categorical PMcore effects name the parameter, covariate, reference level, and
active level. Values must be finite, present, and constant within a subject.
PMcore does not infer or rewrite nonlinear covariate equations.

## Residual error

NONMEM proportional error with `$SIGMA 0.01`:

```text
$ERROR
Y = F + F * EPS(1)
```

uses an EPS standard deviation of `0.1`. The PMcore declaration receives that
coefficient directly:

```rust
.error_model("cp", ResidualErrorModel::proportional(0.1))
```

Combined PMcore error uses additive SD `a` and proportional coefficient `b`:

```rust
.error_model("cp", ResidualErrorModel::combined(0.2, 0.1))
```

Correlated combined error additionally declares `rho` and has

```text
Var(Y | f) = a^2 + 2 rho a b f + b^2 f^2.
```

Each measured output requires its own explicit declaration. PMcore does not use
the data `ErrorPoly` values to select parametric residual scoring.

## Estimation

A typical PMcore fit ends with an explicit configuration:

```rust
let result = problem.fit_with(
    SaemConfig::new()
        .burn_in(100)
        .k1_iterations(300)
        .k2_iterations(200)
        .n_chains(4)
        .seed(42),
)?;
```

The default finite schedule reports `MaxCycles`. Operational `Converged`
termination requires an additional explicit policy. Conditional objectives,
population marginal likelihood, information criteria, and uncertainty retain
distinct result fields and availability statuses.
