//! SAEM Validation Test Module
//!
//! This module contains comprehensive tests to validate the PMcore SAEM implementation
//! against the R saemix reference implementation.
//!
//! # Test Structure
//!
//! 1. **Component Tests**: Verify individual components match R exactly
//!    - Parameter transformations (φ ↔ ψ)
//!    - Sufficient statistics computation
//!    - Step size schedule
//!
//! 2. **Integration Tests**: Verify algorithm phases
//!    - E-step MCMC sampling
//!    - M-step parameter updates
//!
//! 3. **End-to-End Tests**: Compare full algorithm results
//!    - One-compartment IV (synthetic)
//!    - Theophylline (standard reference)
//!
//! # Running Tests
//!
//! ```bash
//! # Run all validation tests
//! cargo test saem_validation -- --nocapture
//!
//! # Run specific test
//! cargo test test_component_transforms -- --nocapture
//! ```

pub mod reference;
pub mod tests;
