//! Non-parametric population structures
//!
//! This module contains structures specific to non-parametric algorithms,
//! which represent population distributions as discrete sets of support points
//! with associated probability weights.
//!
//! # Key Structures
//!
//! - [`Theta`]: Discrete support points representing the joint population parameter distribution
//! - [`Psi`]: Likelihood matrix containing p(y|ψ) for each subject and support point
//! - [`Weights`]: Probability weights associated with each support point

pub mod psi;
pub mod theta;
pub mod weights;

pub use psi::Psi;
pub use theta::Theta;
pub use weights::Weights;
