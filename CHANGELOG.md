# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
