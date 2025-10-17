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
use crate::routines::settings::Settings;
use crate::structs::psi::calculate_psi;
use crate::structs::theta::Theta;

// TODO: Add support for loading and maintenance doses

/// Target type for dose optimization
///
/// Determines whether targets in the "future" file represent concentrations or AUCs.
/// This matches Fortran's ITARGET parameter (1=concentration, 2=AUC).
#[derive(Debug, Clone, Copy)]
pub enum Target {
    /// Target concentrations at observation times (ITARGET=1)
    Concentration,
    /// Target cumulative AUC values from time 0 (ITARGET=2)
    /// AUC is calculated using trapezoidal rule with dense time grid
    AUC,
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
    pub prior_theta: Theta,     // Original population prior support points
    pub prior_weights: Weights, // Original population prior probabilities
    pub theta: Theta,           // Posterior support points (filtered/refined from prior)
    pub posterior: Weights,     // Patient-specific posterior probabilities from NPAGFULL11
    pub target: Subject,
    pub target_type: Target,    // Whether targets are concentrations or AUCs
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
    /// 2. NPAGFULL: Refinement of each filtered point (controlled by max_cycles)
    ///
    /// # Arguments
    /// * `prior_theta` - Prior support points from population NPAG
    /// * `prior_weights` - Prior probabilities (population)
    /// * `past_data` - Patient's historical data (doses and observations). None = use prior directly
    /// * `target` - Future dosing template with target concentrations
    /// * `eq` - Pharmacokinetic/pharmacodynamic model
    /// * `error_models` - Error model specifications
    /// * `doserange` - Allowable dose range
    /// * `bias_weight` - Lambda parameter (0=personalized, 1=population)
    /// * `settings` - Settings for NPAG (used if NPAGFULL refinement is enabled)
    /// * `max_cycles` - Maximum cycles for NPAGFULL refinement (0=skip, 500=Fortran default)
    /// * `target_type` - Whether targets are Concentration or AUC values
    pub fn new(
        prior_theta: &Theta,
        prior_weights: &Weights,
        past_data: Option<Subject>,
        target: Subject,
        eq: ODE,
        error_models: ErrorModels,
        doserange: DoseRange,
        bias_weight: f64,
        settings: Settings,
        max_cycles: usize,
        target_type: Target,
    ) -> Result<Self> {
        // Calculate two-step posterior if past data exists and has observations
        // This matches Fortran logic: IF(INCLUDPAST .EQ. 1 .AND. IPRIOROBS .EQ. 1)
        let (posterior_theta, posterior_weights, final_past_data) = match &past_data {
            None => {
                // INCLUDPAST = 0: No past data provided
                tracing::info!("No past data - using prior density directly");
                (
                    prior_theta.clone(),
                    prior_weights.clone(),
                    Subject::builder("Empty").build(),
                )
            }
            Some(past_subject) => {
                // Check if past data has observations (IPRIOROBS check)
                let has_observations = !past_subject.occasions().is_empty()
                    && past_subject.occasions().iter().any(|occ| {
                        occ.events()
                            .iter()
                            .any(|e| matches!(e, Event::Observation(_)))
                    });

                if !has_observations {
                    // IPRIOROBS = 0: Past data exists but no observations
                    tracing::info!("Past data has no observations - using prior density directly");
                    (
                        prior_theta.clone(),
                        prior_weights.clone(),
                        past_subject.clone(),
                    )
                } else {
                    // INCLUDPAST = 1 AND IPRIOROBS = 1: Calculate posterior from past data
                    tracing::info!("Calculating Bayesian posterior from past data...");
                    let past_data_obj = Data::new(vec![past_subject.clone()]);
                    let (filtered_theta, filtered_weights) = Self::calculate_posterior(
                        prior_theta,
                        prior_weights,
                        &past_data_obj,
                        &eq,
                        &error_models,
                    )?;

                    // Step 2: Optionally refine with NPAGFULL (controlled by max_cycles)
                    let (final_theta, final_weights) =
                        if max_cycles > 0 && filtered_theta.matrix().nrows() > 0 {
                            tracing::info!(
                                "Refining {} filtered points with NPAGFULL (max {} cycles)...",
                                filtered_theta.matrix().nrows(),
                                max_cycles
                            );

                            // Update settings with max_cycles
                            let mut npag_settings = settings.clone();
                            npag_settings.set_cycles(max_cycles);

                            Self::refine_with_npagfull(
                                &filtered_theta,
                                &filtered_weights,
                                &past_data_obj,
                                &eq,
                                &npag_settings,
                            )?
                        } else {
                            if max_cycles == 0 {
                                tracing::info!("Skipping NPAGFULL refinement (max_cycles=0)");
                            }
                            (filtered_theta, filtered_weights)
                        };

                    (final_theta, final_weights, past_subject.clone())
                }
            }
        };

        Ok(Self {
            past_data: final_past_data,
            prior_theta: prior_theta.clone(),
            prior_weights: prior_weights.clone(),
            theta: posterior_theta,
            posterior: posterior_weights,
            target,
            target_type,
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
    /// Implements Bayesian filtering by:
    /// 1. First burke call to get initial posterior probabilities
    /// 2. Lambda filtering with NPAGFULL11-specific 1e-100 threshold
    ///
    /// Note: No QR decomposition or second burke call is performed.
    pub fn calculate_posterior(
        prior_theta: &Theta,
        _prior_weights: &Weights,
        past_data: &Data,
        eq: &ODE,
        error_models: &ErrorModels,
    ) -> Result<(Theta, Weights)> {
        // Calculate psi matrix P(data|theta_i) for all support points
        let psi = calculate_psi(eq, past_data, prior_theta, error_models, false, true)?;

        // First burke call to get initial posterior probabilities
        let (initial_weights, _) = burke(&psi)?;

        // NPAGFULL11 filtering: Keep all points within 1e-100 of the maximum weight
        // This is different from NPAG's condensation - NO QR decomposition here!
        // Fortran: "THOSE WHOSE PROBABILITIES ARE WITHIN 1.D-100 OF THE BEST GRID PT."
        let max_weight = initial_weights
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let threshold = 1e-100; // NPAGFULL11-specific threshold

        let keep_lambda: Vec<usize> = initial_weights
            .iter()
            .enumerate()
            .filter(|(_, lam)| *lam > threshold * max_weight)
            .map(|(i, _)| i)
            .collect();

        // Filter theta to keep only points above threshold
        let mut filtered_theta = prior_theta.clone();
        filtered_theta.filter_indices(&keep_lambda);

        // Filter the weights to keep only those above threshold
        // Use direct indexing instead of iter().nth() to avoid O(n²) complexity
        let filtered_weights: Vec<f64> = keep_lambda.iter().map(|&i| initial_weights[i]).collect();

        // Renormalize the filtered weights to sum to 1
        let sum: f64 = filtered_weights.iter().sum();
        let final_weights = Weights::from_vec(filtered_weights.iter().map(|w| w / sum).collect());

        tracing::info!(
            "NPAGFULL11 complete: {} -> {} support points (lambda filter only, no QR, no second burke)",
            prior_theta.matrix().nrows(),
            filtered_theta.matrix().nrows()
        );

        if filtered_theta.matrix().nrows() > 0 {
            tracing::debug!(
                "Max weight: {:.6e}, threshold: {:.6e}",
                max_weight,
                threshold * max_weight
            );
            tracing::debug!(
                "Kept {} points with weights above threshold",
                keep_lambda.len()
            );
        }

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
        let mut kept_weights: Vec<f64> = Vec::new();
        let num_points = filtered_theta.matrix().nrows();

        for i in 0..num_points {
            tracing::debug!("Refining point {}/{}", i + 1, num_points);

            // Get the current filtered point
            let point: Vec<f64> = filtered_theta.matrix().row(i).iter().copied().collect();

            // Create a single-point theta as starting point for NPAG
            // We need to create a matrix with the correct dimensions (1 row, n_params columns)
            let n_params = point.len();
            let single_point_matrix = Mat::from_fn(1, n_params, |_r, c| point[c]);
            let single_point_theta =
                Theta::from_parts(single_point_matrix, settings.parameters().clone());

            // Create NPAG settings with limited cycles for refinement
            // Set the prior to this single point so get_prior() returns the correct theta
            let mut npag_settings = settings.clone();
            // Cycles are already set by caller via max_cycles parameter
            npag_settings.disable_output(); // Don't write files for each refinement
            npag_settings.set_prior(crate::routines::initialization::Prior::Theta(
                single_point_theta.clone(),
            ));

            // Create and run NPAG
            let mut npag = NPAG::new(npag_settings, eq.clone(), past_data.clone())?;
            // The theta will be initialized from get_prior() automatically, but we set it explicitly here
            npag.set_theta(single_point_theta);

            tracing::debug!(
                "Starting NPAG refinement for point {}/{}",
                i + 1,
                num_points
            );

            // Run NPAG optimization using the standard API
            let refinement_result = npag.initialize().and_then(|_| {
                while !npag.next_cycle()? {}
                Ok(())
            });

            // If refinement failed (e.g., zero probability), skip this point
            if let Err(e) = refinement_result {
                tracing::warn!(
                    "Failed to refine point {}/{}:  {} - skipping",
                    i + 1,
                    num_points,
                    e
                );
                continue;
            }

            tracing::debug!(
                "NPAG converged with {} final point(s)",
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
                kept_weights.push(filtered_weights[i]);
            } else if refined_theta.matrix().nrows() == 1 {
                // Single point - use it
                let refined_point: Vec<f64> =
                    refined_theta.matrix().row(0).iter().copied().collect();
                refined_points.push(refined_point);
                kept_weights.push(filtered_weights[i]);
            } else {
                // Multiple points - this is actually expected with NPAG
                // We take the first point (they're already filtered by condensation)
                let refined_point: Vec<f64> =
                    refined_theta.matrix().row(0).iter().copied().collect();
                refined_points.push(refined_point);
                kept_weights.push(filtered_weights[i]);
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

        // Renormalize the kept weights (in case we skipped some points due to zero probability)
        let weight_sum: f64 = kept_weights.iter().sum();
        let normalized_weights = if weight_sum > 0.0 {
            Weights::from_vec(kept_weights.iter().map(|w| w / weight_sum).collect())
        } else {
            // Fallback to uniform weights if something went wrong
            Weights::uniform(n_points)
        };

        tracing::info!(
            "NPAGFULL refinement complete: {} -> {} refined points",
            filtered_theta.matrix().nrows(),
            refined_theta.matrix().nrows()
        );

        Ok((refined_theta, normalized_weights))
    }

    /// Helper method to run a single optimization with specified weights
    /// Returns (optimal_doses, cost, auc_predictions, predictions)
    fn run_single_optimization(
        &self,
        weights: &Weights,
        method_name: &str,
    ) -> Result<(Vec<f64>, f64, Option<Vec<(f64, f64)>>, NPPredictions)> {
        let min_dose = self.doserange.min;
        let max_dose = self.doserange.max;
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
            "Running {} optimization with {} support points",
            method_name,
            self.theta.matrix().nrows()
        );

        // Make initial simplex
        let initial_guess = (min_dose + max_dose) / 2.0;
        let initial_point = vec![initial_guess; all_doses.len()];
        let initial_simplex = create_initial_simplex(&initial_point);

        // Create a modified problem with the custom weights
        let mut problem_with_weights = self.clone();
        problem_with_weights.posterior = weights.clone();

        // Run optimization
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(initial_simplex);
        let opt = Executor::new(problem_with_weights.clone(), solver)
            .configure(|state| state.max_iters(50))
            .run()?;

        let result = opt.state();
        let optimal_doses = result.best_param.clone().unwrap();
        let final_cost = result.best_cost;

        tracing::info!(
            "{} optimization complete, cost: {:.6}",
            method_name,
            final_cost
        );

        // Calculate AUC predictions if in AUC mode
        let auc_predictions = if matches!(self.target_type, Target::AUC) {
            let obs_times: Vec<f64> = target_subject
                .occasions()
                .iter()
                .flat_map(|occ| occ.events())
                .filter_map(|event| match event {
                    Event::Observation(obs) => Some(obs.time()),
                    _ => None,
                })
                .collect();

            let idelta = self.settings.predictions().idelta;
            let start_time = 0.0;
            let end_time = obs_times.last().copied().unwrap_or(0.0);
            let dense_times = calculate_dense_times(start_time, end_time, &obs_times, idelta);

            let subject_id = target_subject.id().to_string();
            let mut builder = Subject::builder(&subject_id);

            let mut dose_number = 0;
            for occasion in target_subject.occasions() {
                for event in occasion.events() {
                    match event {
                        Event::Bolus(bolus) => {
                            builder = builder.bolus(bolus.time(), optimal_doses[dose_number], 0);
                            dose_number += 1;
                        }
                        Event::Infusion(_) => {
                            tracing::warn!("Infusions not fully supported in AUC mode");
                        }
                        Event::Observation(_) => {}
                    }
                }
            }

            for &t in &dense_times {
                builder = builder.observation(t, -99.0, 0);
            }

            let dense_subject = builder.build();
            let mut mean_aucs = vec![0.0; obs_times.len()];

            for (row, weight) in self.theta.matrix().row_iter().zip(weights.iter()) {
                let spp = row.iter().copied().collect::<Vec<f64>>();
                let pred = self.eq.simulate_subject(&dense_subject, &spp, None)?;
                let dense_concentrations = pred.0.flat_predictions();
                let aucs = calculate_auc_at_times(&dense_times, &dense_concentrations, &obs_times);

                for (i, &auc) in aucs.iter().enumerate() {
                    mean_aucs[i] += weight * auc;
                }
            }

            Some(obs_times.into_iter().zip(mean_aucs.into_iter()).collect())
        } else {
            None
        };

        // Calculate predictions
        let mut target_with_optimal = target_subject.clone();
        let mut dose_number = 0;
        for occasion in target_with_optimal.iter_mut() {
            for event in occasion.iter_mut() {
                match event {
                    Event::Bolus(bolus) => {
                        bolus.set_amount(optimal_doses[dose_number]);
                        dose_number += 1;
                    }
                    Event::Infusion(infusion) => {
                        infusion.set_amount(optimal_doses[dose_number]);
                        dose_number += 1;
                    }
                    Event::Observation(_) => {}
                }
            }
        }

        use faer::Mat;
        let posterior_matrix =
            Mat::from_fn(1, weights.weights().nrows(), |_row, col| *weights.weights().get(col));
        let posterior = Posterior::from(posterior_matrix);

        let preds = NPPredictions::calculate(
            &self.eq,
            &Data::new(vec![target_with_optimal]),
            self.theta.clone(),
            weights,
            &posterior,
            0.0,
            0.0,
        )?;

        Ok((optimal_doses, final_cost, auc_predictions, preds))
    }

    pub fn optimize(self) -> Result<BestDoseResult> {
        let n_points = self.theta.matrix().nrows();
        
        tracing::info!(
            "Starting dual optimization approach (matching Fortran BESTDOS113+)"
        );
        
        // FIRST OPTIMIZATION: Use posterior weights from NPAGFULL11
        tracing::info!("{}", "=".repeat(60));
        tracing::info!("OPTIMIZATION 1: Posterior weights (patient-specific)");
        tracing::info!("{}", "=".repeat(60));
        
        let (doses1, cost1, auc1, preds1) = 
            self.run_single_optimization(&self.posterior, "Posterior")?;
        
        // SECOND OPTIMIZATION: Use uniform weights (population-based)
        tracing::info!("{}", "=".repeat(60));
        tracing::info!("OPTIMIZATION 2: Uniform weights (population)");
        tracing::info!("{}", "=".repeat(60));
        
        let uniform_weights = Weights::uniform(n_points);
        let (doses2, cost2, auc2, preds2) = 
            self.run_single_optimization(&uniform_weights, "Uniform")?;
        
        // Compare and pick the better result
        tracing::info!("{}", "=".repeat(60));
        tracing::info!("COMPARISON:");
        tracing::info!("  Posterior optimization: cost = {:.6}", cost1);
        tracing::info!("  Uniform optimization:   cost = {:.6}", cost2);
        
        
       
        
        let (final_doses, final_cost, final_auc, final_preds, method) = if cost1 <= cost2 {
            tracing::info!("  → Selected: Posterior weights (lower cost)");
            (doses1, cost1, auc1, preds1, "posterior")
        } else {
            tracing::info!("  → Selected: Uniform weights (lower cost)");
            (doses2, cost2, auc2, preds2, "uniform")
        };
        tracing::info!("{}", "=".repeat(60));
        
        Ok(BestDoseResult {
            dose: final_doses,
            objf: final_cost,
            status: "Converged".to_string(),
            preds: final_preds,
            auc_predictions: final_auc,
            optimization_method: method.to_string(),
        })
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

/// Generate dense time grid for AUC calculation
///
/// Creates a rich set of times between start and end, with points at:
/// - Regular intervals of `idelta` minutes
/// - All observation times from `obs_times`
///
/// Times are sorted and deduplicated (within 1e-10 tolerance).
/// This mimics Fortran's CALCTPRED2 subroutine.
///
/// # Arguments
/// * `start` - Starting time (usually 0.0 for "future")
/// * `end` - Ending time (last observation time)
/// * `obs_times` - Observation times to include in the grid
/// * `idelta` - Time interval in minutes for dense sampling
///
/// # Returns
/// Vector of sorted, unique times for simulation
fn calculate_dense_times(start: f64, end: f64, obs_times: &[f64], idelta: f64) -> Vec<f64> {
    let mut times = Vec::new();
    
    // Add regular grid points (idelta in minutes, times in hours)
    let idelta_hours = idelta / 60.0;
    let num_intervals = ((end - start) / idelta_hours).ceil() as usize;
    
    for i in 0..=num_intervals {
        let t = start + (i as f64) * idelta_hours;
        if t <= end {
            times.push(t);
        }
    }
    
    // Add all observation times
    times.extend_from_slice(obs_times);
    
    // Sort
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Deduplicate with tolerance (1e-10 as in Fortran THESAME)
    let tolerance = 1e-10;
    let mut unique_times = Vec::new();
    let mut last_time = f64::NEG_INFINITY;
    
    for &t in &times {
        if (t - last_time).abs() > tolerance {
            unique_times.push(t);
            last_time = t;
        }
    }
    
    unique_times
}

/// Calculate cumulative AUC at target times using trapezoidal rule
///
/// Takes dense concentration predictions and calculates cumulative AUC
/// from the first time point. AUC values at target observation times
/// are extracted and returned.
///
/// # Arguments
/// * `dense_times` - Dense time grid (must include all `target_times`)
/// * `dense_predictions` - Concentration predictions at `dense_times`
/// * `target_times` - Observation times where AUC should be extracted
///
/// # Returns
/// Vector of AUC values at `target_times`
fn calculate_auc_at_times(
    dense_times: &[f64],
    dense_predictions: &[f64],
    target_times: &[f64],
) -> Vec<f64> {
    assert_eq!(dense_times.len(), dense_predictions.len());
    
    let mut target_aucs = Vec::with_capacity(target_times.len());
    let mut auc = 0.0;
    let mut target_idx = 0;
    let tolerance = 1e-10;
    
    for i in 1..dense_times.len() {
        // Update cumulative AUC using trapezoidal rule
        let dt = dense_times[i] - dense_times[i - 1];
        let avg_conc = (dense_predictions[i] + dense_predictions[i - 1]) / 2.0;
        auc += avg_conc * dt;
        
        // Check if current time matches next target time
        if target_idx < target_times.len() {
            if (dense_times[i] - target_times[target_idx]).abs() < tolerance {
                target_aucs.push(auc);
                target_idx += 1;
            }
        }
    }
    
    target_aucs
}

impl CostFunction for BestDoseProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        tracing::info!("Cost function called with dose: {:?}", param);
        
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

        // Build observation vector (target concentrations or AUCs)
        let obs_vec: Vec<f64> = target_subject
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .filter_map(|event| match event {
                Event::Observation(obs) => obs.value(),
                _ => None,
            })
            .collect();

        // Build observation times vector
        let obs_times: Vec<f64> = target_subject
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .filter_map(|event| match event {
                Event::Observation(obs) => Some(obs.time()),
                _ => None,
            })
            .collect();

        let n_obs = obs_vec.len();
        if n_obs == 0 {
            return Err(anyhow::anyhow!("no observations found in target subject"));
        }

        // Check if we have any support points
        if self.theta.matrix().nrows() == 0 {
            return Err(anyhow::anyhow!(
                "No support points in posterior! All points were filtered out during NPAGFULL11. \
                This suggests the past observations are incompatible with the population prior. \
                Check that: (1) the model is correct, (2) the prior ranges include plausible parameter values, \
                (3) the error model is reasonable."
            ));
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
            .zip(self.prior_weights.iter())
        // Prior (population)
        {
            let spp = row.iter().copied().collect::<Vec<f64>>();

            // Get predictions based on target type
            let preds_i: Vec<f64> = match self.target_type {
                Target::Concentration => {
                    // Simulate at observation times only
                    let pred = self.eq.simulate_subject(&target_subject, &spp, None)?;
                    pred.0.flat_predictions()
                }
                Target::AUC => {
                    // For AUC: simulate at dense time grid and calculate cumulative AUC
                    let idelta = self.settings.predictions().idelta;
                    let start_time = 0.0; // Future starts at 0
                    let end_time = obs_times.last().copied().unwrap_or(0.0);
                    
                    // Generate dense time grid
                    let dense_times = calculate_dense_times(start_time, end_time, &obs_times, idelta);
                    
                    // Create temporary subject with dense time points for simulation
                    // We need to rebuild the subject with observations at dense times
                    let subject_id = target_subject.id().to_string();
                    let mut builder = Subject::builder(&subject_id);
                    
                    // Add all doses from original subject
                    for occasion in target_subject.occasions() {
                        for event in occasion.events() {
                            match event {
                                Event::Bolus(bolus) => {
                                    builder = builder.bolus(bolus.time(), bolus.amount(), 0);
                                }
                                Event::Infusion(_infusion) => {
                                    // TODO: Add proper infusion support
                                    // For now, skip infusions in AUC mode
                                    tracing::warn!("Infusions not yet supported in AUC mode");
                                }
                                Event::Observation(_) => {} // Skip original observations
                            }
                        }
                    }
                    
                    // Add observations at dense times (with missing values for timing only)
                    for &t in &dense_times {
                        builder = builder.observation(t, -99.0, 0);
                    }
                    
                    let dense_subject = builder.build();
                    
                    // Simulate at dense times
                    let pred = self.eq.simulate_subject(&dense_subject, &spp, None)?;
                    let dense_predictions = pred.0.flat_predictions();
                    
                    // Calculate AUC at observation times
                    let aucs = calculate_auc_at_times(&dense_times, &dense_predictions, &obs_times);
                    
                    // DEBUG: Print what we're calculating
                    tracing::debug!(
                        "AUC simulation: params={:?}, dense_times={} points, obs_times={:?}, predicted_aucs={:?}",
                        spp, 
                        dense_times.len(),
                        obs_times,
                        aucs
                    );
                    
                    aucs
                }
            };

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
    /// AUC values at observation times (only populated when target_type is AUC)
    pub auc_predictions: Option<Vec<(f64, f64)>>, // (time, auc) pairs
    /// Which optimization method produced the best result: "posterior" or "uniform"
    /// Matches Fortran's dual-optimization approach (BESTDOS113+)
    pub optimization_method: String,
}
