# NPcore
[![Build](https://github.com/LAPKB/NPcore/actions/workflows/rust.yml/badge.svg)](https://github.com/LAPKB/NPcore/actions/workflows/rust.yml)
[![Security Audit](https://github.com/LAPKB/NPcore/actions/workflows/security_audit.yml/badge.svg)](https://github.com/LAPKB/NPcore/actions/workflows/security_audit.yml)

Rust Library with the building blocks needed to create new Non-Parametric algorithms and its integration with [Pmetrics]([https://link-url-here.org](https://github.com/LAPKB/Pmetrics)).

## Implemented functionality

* Solver for ODE-based population pharmacokinetic models
* Supports the Pmetrics data format for seamless integration
* Basic NPAG implementation for parameter estimation
* Covariate support, carry-forward or linear interpolation
* Option to cache results for improved speed
* Powerful simulation engine 
* Informative Terminal User Interface (TUI)


## Examples

There are two examples using NPAG implemented in this repository.

run the following commands to run them:

```
cargo run --example two_eq_lag --release
cargo run --example bimodal_ke --release
```


Look at the corresponding `examples/_example_/*.toml`-file to change the configuration of each run.
