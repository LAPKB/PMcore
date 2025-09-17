use anyhow::{Ok, Result};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::prelude::*;
use crate::routines::output::posterior::Posterior;
use crate::routines::output::predictions::NPPredictions;
use pharmsol::prelude::*;
use pharmsol::{Data, ODE};

use crate::algorithms::npag::burke;
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
    pub target: Subject,
    pub eq: ODE,
    pub doserange: DoseRange,
    pub bias_weight: f64,
    pub error_models: ErrorModels,
}

impl BestDoseProblem {
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

        // Make initial simplex of the Nelder-Mead solver
        let initial_guess = (min_dose + max_dose) / 2.0;
        let initial_point = vec![initial_guess; all_doses.len()];
        let initial_simplex = create_initial_simplex(&initial_point);

        // Initialize the Nelder-Mead solver with correct generic types
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(initial_simplex);
        let problem = self;

        let opt = Executor::new(problem, solver)
            .configure(|state| state.max_iters(1000))
            .run()?;

        let result = opt.state();

        let optimaldose = BestDoseResult {
            dose: result.best_param.clone().unwrap(),
            objf: result.best_cost,
            status: result.termination_status.to_string(),
        };

        Ok(optimaldose)
    }

    pub fn bias(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
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
                        // Set the dose to the new dose
                        bolus.set_amount(param[dose_number]);
                        dose_number += 1;
                    }
                    Event::Infusion(infusion) => {
                        // Set the dose to the new dose
                        infusion.set_amount(param[dose_number]);
                        dose_number += 1;
                    }
                    Event::Observation(_) => {}
                }
            }
        }

        // Calculate psi, in order to determine the optimal weights of the support points in Theta for the target subject
        let psi = calculate_psi(
            &self.eq,
            &Data::new(vec![target_subject.clone()]),
            &self.theta,
            &self.error_models,
            false,
            true,
        )?;

        // Calculate the optimal weights
        let (w, likelihood) = burke(&psi)?;
        tracing::debug!("Likelihood: {}", likelihood);

        // Calculate posterior
        let posterior = Posterior::calculate(&psi, &w)?;

        // Calculate predictions
        let predictions = NPPredictions::calculate(
            &self.eq,
            &Data::new(vec![target_subject.clone()]),
            self.theta.clone(),
            &w,
            &posterior,
            0.0,
            0.0,
        )?;

        // Accumulator for the variance component
        let mut variance = 0.0;

        // Accumulator for the bias component
        let mut bias = 0.0;

        // Iterate over the predictions
        for pred in predictions.predictions() {
            // The squared error of the posterior is added to the variance
            if let Some(squared_error) = pred.obs().map(|obs| (obs - pred.post_mean()).powi(2)) {
                variance += squared_error;
            }

            // The squared error of the population prediction is added to the variance
            if let Some(squared_error) = pred.obs().map(|obs| (obs - pred.pop_mean()).powi(2)) {
                bias += squared_error;
            }
        }

        // Calculate the objective function
        let cost = (1.0 - self.bias_weight) * variance + self.bias_weight * bias;

        // Use the natural logarithm of the cost as the objective function
        Ok(cost.ln())
    }
}

#[derive(Debug)]
pub struct BestDoseResult {
    pub dose: Vec<f64>,
    pub objf: f64,
    pub status: String,
}
