//! Routines for population pharmacokinetic algorithms
//!
//! This module contains common routines shared across algorithms, including
//! data processing, optimization, output generation, and MCMC sampling.

// Routines for condensation (non-parametric)
pub mod condensation;
// Routines for estimation (non-parametric)
pub mod estimation;
// Routines for expansion (non-parametric)
pub mod expansion;
// Routines for initialization
pub mod initialization;
// Routines for logging
pub mod logger;
// Routines for output
pub mod output;
// MCMC sampling routines (parametric)
pub mod sampling;
// Routines for settings
pub mod settings;
