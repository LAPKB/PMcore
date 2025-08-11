# PMcore

[![Build](https://github.com/LAPKB/PMcore/actions/workflows/rust.yml/badge.svg)](https://github.com/LAPKB/PMcore/actions/workflows/rust.yml)
[![Documentation](https://github.com/LAPKB/PMcore/actions/workflows/docs.yml/badge.svg)](https://lapkb.github.io/PMcore/pmcore/)
[![Security Audit](https://github.com/LAPKB/PMcore/actions/workflows/security_audit.yml/badge.svg)](https://github.com/LAPKB/PMcore/actions/workflows/security_audit.yml)
[![crates.io](https://img.shields.io/crates/v/pmcore.svg)](https://crates.io/crates/pmcore)

Rust library with the building blocks to create and implement new non-parametric algorithms for population pharmacokinetic modelling and their integration with [Pmetrics](https://github.com/LAPKB/Pmetrics).

## Implemented functionality

- Solver for ODE-based population pharmacokinetic models
- Supports the Pmetrics data format for seamless integration
- Covariate support, carry-forward or linear interpolation
- Option to cache results for improved speed
- Powerful simulation engine

## Available algorithms

This project aims to implement several algorithms for non-parametric population pharmacokinetic modelling.

- [x] Non Parametric Adaptive Grid (NPAG)
  - [Yamada et al (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7823953/)
  - [Neely et al (2012)](https://pubmed.ncbi.nlm.nih.gov/22722776/)
- [x] Non Parametric Optimal Design (NPOD)
  - [Otalvaro et al (2023)](https://pubmed.ncbi.nlm.nih.gov/36478350/)
  - [Leary et al (2003)](https://www.page-meeting.org/default.asp?abstract=421)
- [ ] Non Parametric Simulated Annealing (NPSA)
  - [Chen et al (2023)](https://arxiv.org/abs/2301.12656)

In the future we also aim to support parametric algorithms, such as the Iterative 2-Stage Bayesian (IT2B)

## Examples

Look at the examples in the `examples` folder to see how to use this library. The examples cover a variety of scenarios.

You may run them with the following command, e.g.

```
cargo run --example bimodal_ke --release
```

## Documentation

For more information on how to use this crate, please review the [documentation](https://lapkb.github.io/PMcore/)
