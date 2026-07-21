# Inter-individual variability

PMcore defines inter-individual variability (IIV) in transformed parameter
space:

```text
phi(P_i) = phi(TVP_i) + eta_i
eta_i ~ Normal(0, Omega)
```

`phi` is selected by the parameter declaration. `TVP_i` includes the population
value and subject-static covariate offsets. Model execution converts the result
back to natural parameter space.

| PMcore declaration | Individual parameter |
| --- | --- |
| `Parameter::real("p")` | `P_i = TVP_i + eta_i` |
| `Parameter::log("p")` | `P_i = TVP_i * exp(eta_i)` |
| `Parameter::logit("p", lower, upper)` | additive eta on the bounded logit scale |
| `Parameter::probit("p", lower, upper)` | additive eta on the bounded probit scale |

`Parameter::with_initial` always receives the natural-scale typical value at
zero eta, kappa, and covariate offsets.

## Additive IIV

NONMEM:

```text
$THETA
(0, 10) ; initial TVP

$OMEGA
4.0     ; variance

$PK
P = THETA(1) + ETA(1)
```

PMcore:

```rust
use pmcore::prelude::*;

let parameter = Parameter::real("p").with_initial(10.0);
let omega = Omega::diagonal_variances([("p", 4.0)]);
```

Both declarations define `P_i = TVP + eta_i` with `eta_i ~ Normal(0, 4)`.
The initial random-effect standard deviation is `2`.

## Log-normal IIV

NONMEM:

```text
$THETA
(0, 5) ; initial TVCL

$OMEGA
0.09   ; variance on the log scale

$PK
CL = THETA(1) * EXP(ETA(1))
```

PMcore:

```rust
let parameter = Parameter::log("cl").with_initial(5.0);
let omega = Omega::diagonal_variances([("cl", 0.09)]);
```

Both define `CL_i = TVCL * exp(eta_CL,i)`. The transformed-space SD is `0.3`.
The corresponding natural-scale coefficient of variation is
`sqrt(exp(0.09) - 1)`, approximately `0.307`.

## Correlated random effects

NONMEM:

```text
$OMEGA BLOCK(2)
0.09
0.01 0.04
```

PMcore:

```rust
let omega = Omega::diagonal_variances([
    ("cl", 0.09),
    ("v", 0.04),
])
.covariance("cl", "v", 0.01);
```

Both initialize

```text
Omega = [[0.09, 0.01],
         [0.01, 0.04]]
```

In PMcore, undeclared covariances are structural zeros. Declare every covariance
that may be estimated.

`Omega::diagonal_standard_deviations` accepts finite positive SDs and squares
them after checking overflow. Legacy `Omega::diagonal` remains variance-based.

## Fixed population values and covariance entries

NONMEM fixes values with `FIX`:

```text
$THETA
(0, 5 FIX)

$OMEGA
0.09 FIX
```

PMcore fixes the population value and variance independently:

```rust
let parameter = Parameter::log("cl")
    .with_initial(5.0)
    .fixed();

let omega = Omega::new().fixed_variance("cl", 0.09);
```

A fixed population value may retain estimated IIV by using `.fixed()` on the
parameter and an estimated `variance` entry in `Omega`.

## Parameters without IIV

A NONMEM parameter has no IIV when its `$PK` expression contains no `ETA` term:

```text
$THETA
(0, 1)

$PK
BASE = THETA(1)
```

The PMcore equivalent is explicit:

```rust
let parameter = Parameter::real("baseline")
    .with_initial(1.0)
    .without_random_effect();
```

The population value may be fixed or estimated. Estimated no-IIV population and
covariate effects use the observation likelihood directly. Their
observed-information covariance and standard errors remain unsupported until
structural observation sensitivities are available.

## Bounded parameters

PMcore can put eta on a bounded logit or probit scale directly:

```rust
let parameter = Parameter::logit("fm", 0.0, 1.0)
    .with_initial(0.20);
let omega = Omega::diagonal_variances([("fm", 0.10)]);
```

This guarantees `0 < FM_i < 1`. A NONMEM model typically writes the inverse
logit transformation explicitly in `$PK`; PMcore stores the bounds and
transformation in the parameter declaration.

## Inter-occasion variability

PMcore `Iov` uses the same variance, covariance, fixedness, and diagonal
constructor semantics as `Omega`. Kappa is additive in transformed parameter
space and indexed by subject and actual occasion. A parameter may have IIV,
IOV, both, or neither.

See [NONMEM and PMcore model declarations](docs/nonmem-comparison.md) for an IOV
example and a broader syntax comparison.

## Numerical safeguards

PMcore requires finite, symmetric, strictly positive-definite covariance
matrices. Updates preserve fixed entries and structural zeros and must not
increase the covariance objective. No jitter, clipping, projection, or matrix
repair is applied.

Covariate raw first and second moments use the same stochastic-approximation
gain. PMcore forms a coherent centered covariance target before applying masks,
local GEM constraints, and any exploration-only displacement cap. Smoothing
does not apply a second covariance gain.
