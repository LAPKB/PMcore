# NPcore
[![Build](https://github.com/LAPKB/NPcore/actions/workflows/rust.yml/badge.svg)](https://github.com/LAPKB/NPcore/actions/workflows/rust.yml)
[![Security Audit](https://github.com/LAPKB/NPcore/actions/workflows/security_audit.yml/badge.svg)](https://github.com/LAPKB/NPcore/actions/workflows/security_audit.yml)
Rust Library with the building blocks needed to create new Non-Parametric algorithms and it's integration with [Pmetrics]([https://link-url-here.org](https://github.com/LAPKB/Pmetrics)).

## Implemented functionality

* Solve ODE-based population pharmacokinetic models
* Basic NPAG implementation
* Supports covariates
* Option to cache results for improved speed


## Examples

There are two examples implemented in this repository. Both of them using NPAG

run the following commands to run them:

```
cargo run --example two_eq_lag --release
cargo run --example bimodal_ke --release
```


Look at the corresponding examples/_example_/*.toml file to change the configuration of each run.
