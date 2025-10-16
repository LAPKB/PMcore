use anyhow::{Ok, Result};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;
use faer::Mat;

use crate::prelude::*;
use crate::routines::output::posterior::Posterior;
use crate::routines::output::predictions::NPPredictions;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;
use pharmsol::{Data, ODE};

use crate::algorithms::npag::burke;
use crate::algorithms::npag::NPAG;
use crate::algorithms::Algorithms;
use crate::routines::condensation::condense_support_points;
use crate::routines::settings::Settings;
use crate::structs::psi::calculate_psi;
use crate::structs::theta::Theta;

// TODO: AUC as a target
// TODO: Add support for loading and maintenance doses

pub enum Target {
    Concentration(f64),
    AUC(f64),
}

#[derive(Debug, Clone)]
pub struct DoseRange {
    min: f64,
    max: f64,
}

impl DoseRange {
    pub fn new(min: f64, max: f64) -> Self {
        DoseRange { min, max }
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Default for DoseRange {
    fn default() -> Self {
        DoseRange {
            min: 0.0,
            max: f64::MAX,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BestDoseProblem {
    pub past_data: Subject,
    pub theta: Theta,
    pub prior: Weights,     // Population probabilities (for bias calculation)
    pub posterior: Weights, // Patient-specific probabilities from NPAGFULL11 (for variance calculation)
    pub target: Subject,
    pub eq: ODE,
    pub doserange: DoseRange,
    pub bias_weight: f64,
    pub error_models: ErrorModels,
    pub settings: Settings, // Settings for NPAG (needed for NPAGFULL refinement)
}

impl BestDoseProblem {
    /// Create a new BestDoseProblem with two-step posterior calculation
    ///
    /// This implements the Fortran BestDose algorithm:
    /// 1. NPAGFULL11: Bayesian filtering of prior based on past data
    /// 2. NPAGFULL: Refinement of each filtered point (optional)
    ///
    /// # Arguments
    /// * `prior_theta` - Prior support points from population NPAG
    /// * `prior_weights` - Prior probabilities (population)
    /// * `past_data` - Patient's historical data (doses and observations)
    /// * `target` - Future dosing template with target concentrations
    /// * `eq` - Pharmacokinetic/pharmacodynamic model
    /// * `error_models` - Error model specifications
    /// * `doserange` - Allowable dose range
    /// * `bias_weight` - Lambda parameter (0=personalized, 1=population)
    /// * `settings` - Settings for NPAG (used if NPAGFULL refinement is enabled)
    /// * `refine_with_npagfull` - If true, refine filtered points with full NPAG optimization
    pub fn new(
        prior_theta: &Theta,
        prior_weights: &Weights,
        past_data: Subject,
        target: Subject,
        eq: ODE,
        error_models: ErrorModels,
        doserange: DoseRange,
        bias_weight: f64,
        settings: Settings,
        refine_with_npagfull: bool,
    ) -> Result<Self> {
        // Calculate two-step posterior if past data exists and has observations
        let (posterior_theta, posterior_weights) = if past_data.occasions().is_empty()
            || past_data.occasions().iter().all(|occ| {
                occ.events()
                    .iter()
                    .all(|e| !matches!(e, Event::Observation(_)))
            }) {
            // No past data or no observations - use prior as-is
            tracing::info!("No past observations - using prior density directly");
            (prior_theta.clone(), prior_weights.clone())
        } else {
            // Step 1: Calculate Bayesian posterior via NPAGFULL11
            tracing::info!("Calculating Bayesian posterior from past data...");
            let past_data_obj = Data::new(vec![past_data.clone()]);
            let (filtered_theta, filtered_weights) = Self::calculate_posterior(
                prior_theta,
                prior_weights,
                &past_data_obj,
                &eq,
                &error_models,
            )?;

            // Step 2: Optionally refine with NPAGFULL
            if refine_with_npagfull && filtered_theta.matrix().nrows() > 0 {
                tracing::info!(
                    "Refining {} filtered points with NPAGFULL...",
                    filtered_theta.matrix().nrows()
                );
                Self::refine_with_npagfull(
                    &filtered_theta,
                    &filtered_weights,
                    &past_data_obj,
                    &eq,
                    &settings,
                )?
            } else {
                (filtered_theta, filtered_weights)
            }
        };

        Ok(Self {
            past_data,
            theta: posterior_theta,
            prior: prior_weights.clone(),
            posterior: posterior_weights,
            target,
            eq,
            doserange,
            bias_weight,
            error_models,
            settings,
        })
    }

    /// NPAGFULL11: Bayesian filtering to get compatible points
    /// This is Step 1 of the two-step posterior
    ///
    /// Follows the Fortran NPAGFULL11 algorithm which calls emint twice:
    /// 1. First burke call to get initial posterior probabilities
    /// 2. Lambda filtering with NPAGFULL11-specific 1e-100 threshold
    /// 3. QR decomposition to remove linearly dependent points
    /// 4. Second burke call to recalculate weights on filtered points
    pub fn calculate_posterior(
        prior_theta: &Theta,
        _prior_weights: &Weights,
        past_data: &Data,
        eq: &ODE,
        error_models: &ErrorModels,
    ) -> Result<(Theta, Weights)> {
        // Calculate psi matrix P(data|theta_i) for all support points
        let psi = calculate_psi(eq, past_data, prior_theta, error_models, false, true)?;

        // First burke call - equivalent to emint with ijob=1 (start)
        let (initial_weights, _) = burke(&psi)?;

        // Use the reusable condensation function with NPAGFULL11-specific 1e-100 threshold
        // This does: lambda filtering + QR decomposition + second burke call
        let (filtered_theta, _filtered_psi, final_weights, _objf) = condense_support_points(
            prior_theta,
            &psi,
            &initial_weights,
            1e-100, // NPAGFULL11-specific threshold (much stricter than NPAG's 1e-3)
            1e-8,   // QR decomposition threshold (standard)
        )?;

        tracing::info!(
            "NPAGFULL11 complete: {} -> {} support points",
            prior_theta.matrix().nrows(),
            filtered_theta.matrix().nrows()
        );

        Ok((filtered_theta, final_weights))
    }

    /// NPAGFULL: Refine each filtered point with full NPAG optimization
    /// This is Step 2 of the two-step posterior
    ///
    /// For each filtered support point from NPAGFULL11, run a full NPAG optimization
    /// starting from that point to get a refined "daughter" point. The NPAGFULL11
    /// probabilities are preserved.
    pub fn refine_with_npagfull(
        filtered_theta: &Theta,
        filtered_weights: &Weights,
        past_data: &Data,
        eq: &ODE,
        settings: &Settings,
    ) -> Result<(Theta, Weights)> {
        let mut refined_points = Vec::new();
        let num_points = filtered_theta.matrix().nrows();

        for i in 0..num_points {
            tracing::debug!("Refining point {}/{}", i + 1, num_points);

            // Get the current filtered point
            let point: Vec<f64> = filtered_theta.matrix().row(i).iter().copied().collect();

            // Create a single-point theta as starting point for NPAG
            // We need to create a matrix with the correct dimensions (1 row, n_params columns)
            let n_params = point.len();
            let single_point_matrix = Mat::from_fn(1, n_params, |_r, c| point[c]);
            let single_point_theta = Theta::from_parts(single_point_matrix, settings.parameters().clone());

            // Create NPAG settings with limited cycles for refinement
            // Set the prior to this single point so get_prior() returns the correct theta
            let mut npag_settings = settings.clone();
            npag_settings.set_cycles(100); // Limit cycles for individual refinements
            npag_settings.disable_output(); // Don't write files for each refinement
            npag_settings.set_prior(crate::routines::initialization::Prior::Theta(single_point_theta.clone()));

            // Create and run NPAG
            let mut npag = NPAG::new(npag_settings, eq.clone(), past_data.clone())?;
            // The theta will be initialized from get_prior() automatically, but we set it explicitly here
            npag.set_theta(single_point_theta);

            tracing::debug!("Starting NPAG refinement for point {}/{}", i + 1, num_points);
            
            // Run NPAG optimization loop
            // We need at least 2 cycles to trigger expansion (expansion happens when cycle > 1)
            // This ensures we get the "daughter points" mentioned in Fortran documentation
            let mut cycle_count = 0;
            const MIN_CYCLES: usize = 2; // Force at least 2 cycles to ensure expansion happens
            
            while !npag.converged() || cycle_count < MIN_CYCLES {
                npag.inc_cycle();
                cycle_count += 1;
                
                // Expansion happens first (after cycle 1)
                if cycle_count > 1 {
                    npag.expansion()?;
                }
                
                npag.evaluation()?;
                npag.condensation()?;
                npag.optimizations()?;
                npag.convergence_evaluation();
            }
            
            tracing::debug!(
                "NPAG converged after {} cycles with {} final point(s)",
                cycle_count,
                npag.theta().matrix().nrows()
            );

            // Extract the refined point(s)
            // After NPAG convergence, theta may have multiple points
            // Take the one with highest posterior probability
            let refined_theta = npag.theta();

            if refined_theta.matrix().nrows() == 0 {
                tracing::warn!(
                    "NPAG refinement produced no points for point {}, using original",
                    i + 1
                );
                let original_point: Vec<f64> =
                    filtered_theta.matrix().row(i).iter().copied().collect();
                refined_points.push(original_point);
            } else if refined_theta.matrix().nrows() == 1 {
                // Single point - use it
                let refined_point: Vec<f64> =
                    refined_theta.matrix().row(0).iter().copied().collect();
                refined_points.push(refined_point);
            } else {
                // Multiple points - this is actually expected with NPAG
                // We take the first point (they're already filtered by condensation)
                let refined_point: Vec<f64> =
                    refined_theta.matrix().row(0).iter().copied().collect();
                refined_points.push(refined_point);
                tracing::debug!(
                    "NPAG produced {} points, using first",
                    refined_theta.matrix().nrows()
                );
            }
        }

        // Build refined theta matrix with proper parameters
        let n_params = settings.parameters().len();
        let n_points = refined_points.len();
        let refined_matrix = Mat::from_fn(n_points, n_params, |r, c| refined_points[r][c]);
        let refined_theta = Theta::from_parts(refined_matrix, settings.parameters().clone());

        // Preserve the NPAGFULL11 weights (don't recalculate!)
        tracing::info!(
            "NPAGFULL refinement complete: {} refined points",
            refined_theta.matrix().nrows()
        );

        Ok((refined_theta, filtered_weights.clone()))
    }

    pub fn optimize(self) -> Result<BestDoseResult> {
        let min_dose = self.doserange.min;
        let max_dose = self.doserange.max;

        // Get the target subject
        let target_subject = self.target.clone();

        // Get all dose amounts as a vector
        let all_doses: Vec<f64> = target_subject
            .iter()
            .flat_map(|occ| {
                occ.iter().filter_map(|event| match event {
                    Event::Bolus(bolus) => Some(bolus.amount()),
                    Event::Infusion(infusion) => Some(infusion.amount()),
                    Event::Observation(_) => None,
                })
            })
            .collect();

        tracing::info!(
            "Optimizing {} dose(s) using {} support points (posterior from NPAGFULL11)",
            all_doses.len(),
            self.theta.matrix().nrows()
        );

        // Make initial simplex of the Nelder-Mead solver
        let initial_guess = (min_dose + max_dose) / 2.0;
        let initial_point = vec![initial_guess; all_doses.len()];
        let initial_simplex = create_initial_simplex(&initial_point);

        tracing::debug!("Initial simplex: {:?}", initial_simplex);

        // Initialize the Nelder-Mead solver with correct generic types
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(initial_simplex);
        let problem = self;

        let opt = Executor::new(problem.clone(), solver)
            .configure(|state| state.max_iters(1000))
            .run()?;

        let result = opt.state();

        tracing::info!("Optimization complete, final cost: {:.6}", result.best_cost);

        let preds = {
            // Modify the target subject with the new dose(s)
            let mut target_subject = target_subject.clone();
            let mut dose_number = 0;

            for occasion in target_subject.iter_mut() {
                for event in occasion.iter_mut() {
                    match event {
                        Event::Bolus(bolus) => {
                            // Set the dose to the new dose
                            bolus.set_amount(result.best_param.clone().unwrap()[dose_number]);
                            dose_number += 1;
                        }
                        Event::Infusion(infusion) => {
                            // Set the dose to the new dose
                            infusion.set_amount(result.best_param.clone().unwrap()[dose_number]);
                            dose_number += 1;
                        }
                        Event::Observation(_) => {}
                    }
                }
            }

            // Calculate psi for optimal doses to get final predictions
            let psi = calculate_psi(
                &problem.eq,
                &Data::new(vec![target_subject.clone()]),
                &problem.theta.clone(),
                &problem.error_models,
                false,
                true,
            )?;

            // Use the preserved posterior weights from NPAGFULL11
            // NOT recalculated weights from burke!
            let w = &problem.posterior;

            // Calculate posterior for predictions
            let posterior = Posterior::calculate(&psi, w)?;

            // Calculate predictions using NPAGFULL11 posterior weights
            NPPredictions::calculate(
                &problem.eq,
                &Data::new(vec![target_subject.clone()]),
                problem.theta.clone(),
                w,
                &posterior,
                0.0,
                0.0,
            )?
        };

        let optimaldose = BestDoseResult {
            dose: result.best_param.clone().unwrap(),
            objf: result.best_cost,
            status: result.termination_status.to_string(),
            preds,
        };

        Ok(optimaldose)
    }

    /// Set the bias weight (lambda parameter)
    ///
    /// - λ = 0.0 (default): Full personalization (minimize patient-specific variance)
    /// - λ = 0.5: Balanced between individual and population
    /// - λ = 1.0: Population-based (minimize deviation from population mean)
    pub fn with_bias_weight(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
    }

    /// Legacy alias for with_bias_weight
    pub fn bias(self, weight: f64) -> Self {
        self.with_bias_weight(weight)
    }
}

fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.008;

    // Initialize a Vec to store the vertices of the simplex
    let mut vertices = Vec::new();

    // Add the initial point to the vertices
    vertices.push(initial_point.to_vec());

    // Calculate perturbation values for each component
    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025 // Special case for components equal to 0
        } else {
            perturbation_percentage * initial_point[i]
        };

        let mut perturbed_point = initial_point.to_owned();
        perturbed_point[i] += perturbation;
        vertices.push(perturbed_point);
    }

    vertices
}

impl CostFunction for BestDoseProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        // Modify the target subject with the new dose(s)
        let mut target_subject = self.target.clone();
        let mut dose_number = 0;

        for occasion in target_subject.iter_mut() {
            for event in occasion.iter_mut() {
                match event {
                    Event::Bolus(bolus) => {
                        bolus.set_amount(param[dose_number]);
                        dose_number += 1;
                    }
                    Event::Infusion(infusion) => {
                        infusion.set_amount(param[dose_number]);
                        dose_number += 1;
                    }
                    Event::Observation(_) => {}
                }
            }
        }

        // Build observation vector (target concentrations)
        let obs_vec: Vec<f64> = target_subject
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .filter_map(|event| match event {
                Event::Observation(obs) => obs.value(),
                _ => None,
            })
            .collect();

        let n_obs = obs_vec.len();
        if n_obs == 0 {
            return Err(anyhow::anyhow!("no observations found in target subject"));
        }

        // Accumulators
        let mut variance = 0.0_f64; // Expected squared error E[(target - pred)²]
        let mut y_bar = vec![0.0_f64; n_obs]; // Population mean predictions

        // CRITICAL: Use preserved posterior probabilities from NPAGFULL11 for variance
        // and prior probabilities for bias calculation (population mean)
        // This matches Fortran implementation exactly!

        for ((row, post_prob), prior_prob) in self
            .theta
            .matrix()
            .row_iter()
            .zip(self.posterior.iter()) // Posterior from NPAGFULL11 (patient-specific)
            .zip(self.prior.iter())
        // Prior (population)
        {
            let spp = row.iter().copied().collect::<Vec<f64>>();

            // Simulate the target subject with this support point
            let pred = self.eq.simulate_subject(&target_subject, &spp, None)?;

            // Get per-observation predictions in the same order as obs_vec
            let preds_i: Vec<f64> = pred.0.flat_predictions();

            if preds_i.len() != n_obs {
                return Err(anyhow::anyhow!(
                    "prediction length ({}) != observation length ({})",
                    preds_i.len(),
                    n_obs
                ));
            }

            // Calculate variance term: weighted by POSTERIOR probability
            let mut sumsq_i = 0.0_f64;
            for (j, &obs_val) in obs_vec.iter().enumerate() {
                let pj = preds_i[j];
                let se = (obs_val - pj).powi(2);
                sumsq_i += se;
                // Calculate population mean using PRIOR probabilities
                y_bar[j] += prior_prob * pj;
            }

            variance += post_prob * sumsq_i; // Weighted by posterior
        }

        // Calculate bias term: squared difference from population mean
        let mut bias = 0.0_f64;
        for (j, &obs_val) in obs_vec.iter().enumerate() {
            bias += (obs_val - y_bar[j]).powi(2);
        }

        // Final cost: (1-λ)×Variance + λ×Bias²
        // λ=0: Full personalization (minimize variance)
        // λ=1: Population-based (minimize bias from population)
        let cost = (1.0 - self.bias_weight) * variance + self.bias_weight * bias;

        Ok(cost)
    }
}

#[derive(Debug)]
pub struct BestDoseResult {
    pub dose: Vec<f64>,
    pub objf: f64,
    pub status: String,
    pub preds: NPPredictions,
}
