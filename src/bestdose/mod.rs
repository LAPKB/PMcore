use anyhow::{Ok, Result};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::prelude::*;
use crate::routines::output::posterior::Posterior;
use crate::routines::output::predictions::NPPredictions;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;
use pharmsol::{Data, ODE};

use crate::algorithms::npag::burke;
use crate::structs::psi::calculate_psi;
use crate::structs::theta::Theta;

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
    pub prior: Weights, // These are posterior weights from past_data, fixed during optimization
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

        println!("Initial simplex: {:?}", initial_simplex);

        // Initialize the Nelder-Mead solver
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(initial_simplex);
        let problem = self;

        let opt = Executor::new(problem.clone(), solver)
            .configure(|state| state.max_iters(1000))
            .run()?;

        let result = opt.state();

        let preds = {
            // Modify the target subject with the optimal dose(s)
            let mut target_subject = target_subject.clone();
            let mut dose_number = 0;

            for occasion in target_subject.iter_mut() {
                for event in occasion.iter_mut() {
                    match event {
                        Event::Bolus(bolus) => {
                            bolus.set_amount(result.best_param.clone().unwrap()[dose_number]);
                            dose_number += 1;
                        }
                        Event::Infusion(infusion) => {
                            infusion.set_amount(result.best_param.clone().unwrap()[dose_number]);
                            dose_number += 1;
                        }
                        Event::Observation(_) => {}
                    }
                }
            }

            // Calculate psi for final predictions with optimal doses
            let psi = calculate_psi(
                &problem.eq,
                &Data::new(vec![target_subject.clone()]),
                &problem.theta.clone(),
                &problem.error_models,
                false,
                true,
            )?;

            // Calculate the optimal weights for final output
            let (w, _likelihood) = burke(&psi)?;

            // Calculate posterior
            let posterior = Posterior::calculate(&psi, &w)?;

            // Calculate predictions
            NPPredictions::calculate(
                &problem.eq,
                &Data::new(vec![target_subject.clone()]),
                problem.theta.clone(),
                &w,
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

    pub fn bias(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
    }
}

fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.008;

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());

    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025
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
        // Modify the target subject with the candidate dose(s)
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

        // Build observation vector
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
        let mut variance = 0.0_f64;
        let mut y_bar = vec![0.0_f64; n_obs];

        // Iterate over each support point with FIXED prior probabilities
        // (which are actually posterior from past_data, but fixed during optimization)
        for (row, prob) in self.theta.matrix().row_iter().zip(self.prior.iter()) {
            let spp = row.iter().copied().collect::<Vec<f64>>();

            // Simulate the target subject with this support point
            let pred = self.eq.simulate_subject(&target_subject, &spp, None)?;

            // Get per-observation predictions
            let preds_i: Vec<f64> = pred.0.flat_predictions();

            if preds_i.len() != n_obs {
                return Err(anyhow::anyhow!(
                    "prediction length ({}) != observation length ({})",
                    preds_i.len(),
                    n_obs
                ));
            }

            // Calculate squared prediction error for this support point
            let mut sumsq_i = 0.0_f64;
            for (j, &obs_val) in obs_vec.iter().enumerate() {
                let pj = preds_i[j];
                let se = (obs_val - pj).powi(2);
                sumsq_i += se;
                y_bar[j] += prob * pj; // Weighted mean using fixed probabilities
            }

            variance += prob * sumsq_i; // Expected variance using fixed probabilities
        }

        // Calculate bias
        let mut bias = 0.0_f64;
        for (j, &obs_val) in obs_vec.iter().enumerate() {
            bias += (obs_val - y_bar[j]).powi(2);
        }

        // Final cost
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
