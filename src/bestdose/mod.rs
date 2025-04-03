use anyhow::{Ok, Result};
use argmin::core::TerminationReason;
use argmin::core::TerminationStatus;
use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;

use pharmsol::prelude::*;
use pharmsol::{Data, ODE};

use crate::algorithms::npag::burke;

use crate::structs::psi::calculate_psi;
use crate::structs::theta::Theta;

pub struct DoseOptimizer {
    pub data: Data,
    pub theta: Theta,
    pub target_concentration: f64,
    pub target_time: f64,
    pub eq: ODE,
    pub min_dose: f64,
    pub max_dose: f64,
}

impl CostFunction for DoseOptimizer {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        println!("Evaluating dose: {:?}", param);

        let errmod = pharmsol::ErrorModel::new((0.0, 0.1, 0.0, 0.0), 0.0, &ErrorType::Add);

        let psi = calculate_psi(&self.eq, &self.data, &self.theta, &errmod, false, true);

        let (w, objf) = burke(&psi)?;

        Ok(param + 5.0) // Example cost function
    }
}

pub struct OptimalDose {
    dose: f64,
    cost: f64,
    status: String,
}

pub fn optimize_dose(problem: DoseOptimizer) -> Result<OptimalDose> {
    let min_dose = problem.min_dose;
    let max_dose = problem.max_dose;

    let solver = BrentOpt::new(min_dose, max_dose); // With the given contraints

    let opt = Executor::new(problem, solver)
        .configure(|state| state.max_iters(1000))
        .run()?;

    let result = opt.state();

    match &result.termination_status {
        TerminationStatus::Terminated(status) => match status {
            TerminationReason::SolverConverged => {
                println!("Solver converged");
            }
            _ => {
                println!("Solver terminated with reason: {}", status.text());
            }
        },
        TerminationStatus::NotTerminated => {
            println!("Solver did not terminate");
        }
    }

    let optimaldose = OptimalDose {
        dose: result.param.unwrap(),
        cost: result.cost,
        status: result.termination_status.to_string(),
    };

    println!("Optimal dose: {}", optimaldose.dose);
    println!("Cost: {}", optimaldose.cost);
    println!("Status: {}", optimaldose.status);

    Ok(optimaldose)
}
