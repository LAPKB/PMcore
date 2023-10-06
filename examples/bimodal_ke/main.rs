use std::collections::HashMap;

use eyre::Result;
use npcore::prelude::{
    datafile::{CovLine, Infusion, Scenario},
    predict::{Engine, Predict},
    start,
};
use ode_solvers::*;

const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

#[derive(Debug, Clone)]
struct Model {
    ke: f64,
    v: f64,
    _scenario: Scenario,
    infusions: Vec<Infusion>,
    cov: Option<HashMap<String, CovLine>>,
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let ke = self.ke;

        let _lag = 0.0;

        let mut rateiv = [0.0];
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] += infusion.amount / infusion.dur;
            }
        }

        ///////////////////// USER DEFINED ///////////////

        dy[0] = -ke * y[0] + rateiv[0];

        //////////////// END USER DEFINED ////////////////
    }
}

#[derive(Debug, Clone)]
struct Ode {}

impl<'a> Predict<'a> for Ode {
    type Model = Model;
    type State = State;
    fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> Self::Model {
        Model {
            ke: params[0],
            v: params[1],
            _scenario: scenario,
            infusions: vec![],
            cov: None,
        }
    }
    fn get_output(&self, x: &Self::State, system: &Self::Model, outeq: usize) -> f64 {
        let v = system.v;
        match outeq {
            1 => x[0] / v,
            _ => panic!("Invalid output equation"),
        }
    }
    fn initial_state(&self) -> State {
        State::default()
    }
    fn add_infusion(&self, mut system: Self::Model, infusion: Infusion) -> Model {
        system.infusions.push(infusion);
        system
    }
    fn add_covs(&self, mut system: Self::Model, cov: Option<HashMap<String, CovLine>>) -> Model {
        system.cov = cov;
        system
    }
    fn add_dose(&self, mut state: Self::State, dose: f64, compartment: usize) -> Self::State {
        state[compartment] += dose;
        state
    }
    fn state_step(
        &self,
        mut x: Self::State,
        system: Self::Model,
        time: f64,
        next_time: f64,
    ) -> State {
        let mut stepper = Dopri5::new(system, time, next_time, 1e-3, x, RTOL, ATOL);
        let _res = stepper.integrate();
        let y = stepper.y_out();
        x = *y.last().unwrap();
        x
    }
}

fn main() -> Result<()> {
    let _result = start(
        Engine::new(Ode {}),
        "examples/bimodal_ke/config.toml".to_string(),
    )?;

    Ok(())
}
