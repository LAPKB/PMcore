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

type State = Vector1<f64>;
type Time = f64;
#[derive(Debug, Clone)]
struct Model {
    params: HashMap<String, f64>,
    _scenario: Scenario,
    infusions: Vec<Infusion>,
    cov: Option<HashMap<String, CovLine>>,
}
impl Model {
    pub fn get_param(&self, str: &str) -> f64 {
        *self.params.get(str).unwrap()
    }
}

impl ode_solvers::System<State> for Model {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let ke = self.get_param("ke");

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
    fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario) {
        let params = HashMap::from([("ke".to_string(), params[0]), ("v".to_string(), params[1])]);
        (
            Model {
                params,
                _scenario: scenario.clone(), //TODO remove
                infusions: vec![],
                cov: None,
            },
            scenario.reorder_with_lag(vec![(0.0, 1)]),
        )
    }
    fn get_output(&self, _time: f64, x: &Self::State, system: &Self::Model, outeq: usize) -> f64 {
        let v = system.get_param("v");
        match outeq {
            1 => x[0] / v,
            _ => panic!("Invalid output equation"),
        }
    }
    fn initial_state(&self) -> State {
        State::default()
    }
    fn add_infusion(&self, system: &mut Self::Model, infusion: Infusion) {
        system.infusions.push(infusion);
    }
    fn add_covs(&self, system: &mut Self::Model, cov: Option<HashMap<String, CovLine>>) {
        system.cov = cov;
    }
    fn add_dose(&self, state: &mut Self::State, dose: f64, compartment: usize) {
        state[compartment] += dose;
    }
    fn state_step(&self, x: &mut Self::State, system: &Self::Model, time: f64, next_time: f64) {
        if time >= next_time {
            panic!("time error")
        }
        let mut stepper = Dopri5::new(system.clone(), time, next_time, 1e-3, *x, RTOL, ATOL);
        let _res = stepper.integrate();
        let y = stepper.y_out();
        *x = *y.last().unwrap();
    }
}

fn main() -> Result<()> {
    let _result = start(
        Engine::new(Ode {}),
        "examples/bimodal_ke/config.toml".to_string(),
    )?;

    Ok(())
}
