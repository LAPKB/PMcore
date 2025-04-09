use anyhow::{Ok, Result};
use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;

use pharmsol::prelude::*;
use pharmsol::Equation;
use pharmsol::Predictions;
use pharmsol::{Data, ODE};

use crate::algorithms::npag::burke;
use crate::structs::psi::calculate_psi;
use crate::structs::theta::Theta;

pub enum Target {
    Concentration(f64),
    AUC(f64),
}

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

pub struct BestDoseProblem {
    pub past_data: Data,
    pub theta: Theta,
    pub target_concentration: f64,
    pub target_time: f64,
    pub eq: ODE,
    pub doserange: DoseRange,
    pub bias_weight: f64,
}

impl BestDoseProblem {
    pub fn optimize(self) -> Result<BestDoseResult> {
        let min_dose = self.doserange.min;
        let max_dose = self.doserange.max;

        let solver = BrentOpt::new(min_dose, max_dose);

        let problem = self;

        let opt = Executor::new(problem, solver)
            .configure(|state| state.max_iters(1000))
            .run()?;

        let result = opt.state();

        let optimaldose = BestDoseResult {
            dose: result.param.unwrap(),
            objf: result.cost,
            status: result.termination_status.to_string(),
        };

        Ok(optimaldose)
    }
}

impl CostFunction for BestDoseProblem {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let dose = param.clone();
        let target_subject = Subject::builder("target")
            .bolus(0.0, dose, 0)
            .observation(self.target_time, self.target_concentration, 0)
            .build();

        let errmod = pharmsol::ErrorModel::new((0.0, 0.1, 0.0, 0.0), 0.0, &ErrorType::Add);

        let psi = calculate_psi(&self.eq, &self.past_data, &self.theta, &errmod, false, true);

        let (w, _) = burke(&psi)?;

        // Normalize W to sum to 1
        let w_sum: f64 = w.iter().sum();
        let w: Vec<f64> = w.iter().map(|&x| x / w_sum).collect();

        // Then calcualte the bias

        // Store the mean of the predictions
        // TODO: This needs to handle more than one target
        let mut y_bar = 0.0;

        // Accumulator for weighted sum
        let mut wt_sum = 0.0;
        for (row, prob) in self.theta.matrix().row_iter().zip(w.iter()) {
            let spp = row.iter().copied().collect::<Vec<f64>>();
            let pred = self.eq.simulate_subject(&target_subject, &spp, None);
            wt_sum += pred.0.squared_error() * prob;

            y_bar += pred.0.flat_predictions().first().unwrap() * prob;
        }

        // Bias is the squared difference between the target concentration and the mean of the predictions
        let bias = (y_bar - self.target_concentration).powi(2);

        // Calculate the objective function
        let objf = (1.0 - self.bias_weight) * wt_sum + self.bias_weight * bias;

        // TODO: Repeat with D_flat, and return the best

        Ok(objf) // Example cost function
    }
}

#[derive(Debug)]
pub struct BestDoseResult {
    pub dose: f64,
    pub objf: f64,
    pub status: String,
}
