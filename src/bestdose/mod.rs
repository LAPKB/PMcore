//! # BestDose: dose forecasting and optimization
//!
//! BestDose finds dosing regimens that hit target drug concentrations or AUC
//! values for a given distribution over model parameters.
//!
//! The distribution is supplied by the caller as support points
//! ([`Theta`](crate::estimation::nonparametric::Theta)) and probability
//! [`Weights`](crate::estimation::nonparametric::Weights). It typically comes from a
//! population fit, optionally
//! updated to a patient-specific posterior with the NCNPAG or NPMAP algorithms.
//!
//! # Flow
//!
//! ```rust,no_run,ignore
//! use pmcore::bestdose::{BestDoseProblem, BestDoseOptions, DoseRange, Target};
//! use pmcore::prelude::*;
//!
//! # fn example(eq: pharmsol::prelude::ODE, pop_data: pharmsol::prelude::Data,
//! #            prior_theta: pmcore::estimation::nonparametric::Theta,
//! #            ems: pmcore::AssayErrorModels,
//! #            past_data: Option<pharmsol::prelude::Subject>,
//! #            target: pharmsol::prelude::Subject) -> anyhow::Result<()> {
//! // 1. Fit the population model with any algorithm.
//! let fit = EstimationProblem::nonparametric(eq.clone(), pop_data, prior_theta, ems.clone())?
//!     .fit_with(NpagConfig::default())?;
//!
//! // 2. Choose the distribution: patient-specific posterior (past data) or population.
//! let (theta, weights) = match past_data {
//!     Some(past) => {
//!         let post = EstimationProblem::nonparametric(
//!                 eq.clone(), data::Data::new(vec![past]), fit.get_theta().clone(), ems.clone())?
//!             .fit_with(NcnpagConfig::default())?; // or NpmapConfig::default()
//!         (post.get_theta().clone(), post.weights().clone())
//!     }
//!     None => (fit.get_theta().clone(), fit.weights().clone()),
//! };
//!
//! // 3. Optimize doses.
//! let problem = BestDoseProblem::new(eq, theta, weights)?;
//! let result = problem.optimize(
//!     target,
//!     Target::Concentration,
//!     DoseRange::new(0.0, 300.0),
//!     0.5, // bias λ: 0 = personalized, 1 = population-typical
//!     BestDoseOptions::default(),
//! )?;
//!
//! let optimal_subject = result.subject();
//! let cost = result.cost();
//! # Ok(())
//! # }
//! ```
//!
//! # Cost function
//!
//! `optimize` minimizes, over the optimizable doses, a hybrid objective computed
//! from the single distribution `(theta, weights)`:
//!
//! ```text
//! Cost = (1-λ) × Variance + λ × Bias²
//! Variance = Σᵢ wᵢ Σⱼ (targetⱼ − pred[i,j])²      (expected squared error)
//! Bias²    = Σⱼ (targetⱼ − Σᵢ wᵢ pred[i,j])²       (error of the weighted mean)
//! ```

pub mod cost;
mod optimization;
pub mod predictions;
mod types;

pub use types::{Achievement, BestDoseOptions, BestDoseProblem, BestDoseResult, DoseRange, Target};
