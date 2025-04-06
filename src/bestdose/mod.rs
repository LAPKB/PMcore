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

pub struct DoseOptimizer {
    pub past_data: Data,
    pub theta: Theta,
    pub target_concentration: f64,
    pub target_time: f64,
    pub eq: ODE,
    pub doserange: DoseRange,
    pub bias_weight: f64,
}

impl CostFunction for DoseOptimizer {
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

        let nspp = self.theta.matrix().nrows();
        let bias_factor = 1.0 / (nspp as f64);

        // Calculate BIAS
        let mut bias = 0.0;

        for row in self.theta.matrix().row_iter() {
            let spp = row.iter().copied().collect::<Vec<f64>>();
            let squared_error = self
                .eq
                .simulate_subject(&target_subject, &spp, None)
                .0
                .squared_error();

            bias += squared_error * bias_factor;
        }

        // Calculate the weighted sum
        let mut wt_sum = 0.0;

        for (row, prob) in self.theta.matrix().row_iter().zip(w.iter()) {
            let spp = row.iter().copied().collect::<Vec<f64>>();
            let squared_error = self
                .eq
                .simulate_subject(&target_subject, &spp, None)
                .0
                .squared_error();

            wt_sum += squared_error * prob;
        }

        let objf = (1.0 - self.bias_weight) * wt_sum + self.bias_weight * bias;

        Ok(objf) // Example cost function
    }
}

#[derive(Debug)]
pub struct OptimalDose {
    pub dose: f64,
    pub objf: f64,
    pub status: String,
}

pub fn optimize_dose(problem: DoseOptimizer) -> Result<OptimalDose> {
    let min_dose = problem.doserange.min;
    let max_dose = problem.doserange.max;

    let solver = BrentOpt::new(min_dose, max_dose); // With the given contraints

    let opt = Executor::new(problem, solver)
        .configure(|state| state.max_iters(1000))
        .run()?;

    let result = opt.state();

    let optimaldose = OptimalDose {
        dose: result.param.unwrap(),
        objf: result.cost,
        status: result.termination_status.to_string(),
    };

    Ok(optimaldose)
}
