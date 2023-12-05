#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use std::collections::HashMap;

use eyre::Result;
use npcore::prelude::{
    datafile,
    datafile::{CovLine, Infusion, Scenario},
    predict::{Engine, Predict},
    settings, start,
};
use ode_solvers::*;

const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

type State = SVector<f64, 3>;
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
    fn system(&self, t: Time, x: &State, dx: &mut State) {
        // let ke = self.get_param("ke");
        let ka = self.get_param("ka");
        let vmax0 = self.get_param("vmax0");
        let km = self.get_param("km");
        let vc0 = self.get_param("vc0");
        let fa1 = self.get_param("fa1");
        let kcp = self.get_param("kcp");
        let kpc = self.get_param("kpc");
        let hill = self.get_param("hill");
        let age50 = self.get_param("age50");
        let cov = self.cov.clone().unwrap();
        let wt = cov.get("wt").unwrap().interp(t);
        let ast = cov.get("ast").unwrap().interp(t);
        let alt = cov.get("alt").unwrap().interp(t);
        let age = cov.get("age").unwrap().interp(t);

        let vm = vmax0 * wt.powf(0.75);
        let v = vc0 * wt;
        let fage = age.powf(hill) / (age50.powf(hill) + age.powf(hill));

        let mut rateiv = [0.0]; //TODO: hardcoded
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] = infusion.amount / infusion.dur;
            }
        }
        dx[0] = -ka * x[0];
        dx[1] =
            ka * x[0] + rateiv[0] - fage * vm / (km * v + x[1]) * x[1] - kcp * x[1] + kpc * x[2];
        dx[2] = kcp * x[1] - kpc * x[2];
    }
}

#[derive(Debug, Clone)]
struct Ode {}

impl Predict<'_> for Ode {
    type Model = Model;
    type State = State;
    fn initial_system(&self, parameters: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario) {
        let mut params = HashMap::new();
        params.insert("ka".to_string(), parameters[0].clone());
        params.insert("vmax0".to_string(), parameters[1].clone());
        params.insert("km".to_string(), parameters[2].clone());
        params.insert("vc0".to_string(), parameters[3].clone());
        params.insert("fa1".to_string(), parameters[4].clone());
        params.insert("kcp".to_string(), parameters[5].clone());
        params.insert("kpc".to_string(), parameters[6].clone());
        params.insert("hill".to_string(), parameters[7].clone());
        params.insert("age50".to_string(), parameters[8].clone());
        let system = Model {
            params,
            _scenario: scenario.clone(), //TODO remove
            infusions: vec![],
            cov: None,
        };

        (
            system, // scenario.reorder_with_lag(vec![(0.0, 1)]))
            scenario,
        )
    }
    fn get_output(&self, t: f64, x: &Self::State, system: &Self::Model, outeq: usize) -> f64 {
        let ka = system.get_param("ka");
        let vmax0 = system.get_param("vmax0");
        let km = system.get_param("km");
        let vc0 = system.get_param("vc0");
        let fa1 = system.get_param("fa1");
        let kcp = system.get_param("kcp");
        let kpc = system.get_param("kpc");
        let hill = system.get_param("hill");
        let age50 = system.get_param("age50");
        let cov = system.cov.clone().unwrap();
        let wt = cov.get("wt").unwrap().interp(t);
        let ast = cov.get("ast").unwrap().interp(t);
        let alt = cov.get("alt").unwrap().interp(t);
        let age = cov.get("age").unwrap().interp(t);

        let vm = vmax0 * wt.powf(0.75);
        let v = vc0 * wt;
        let fage = age.powf(hill) / (age50.powf(hill) + age.powf(hill));

        match outeq {
            1 => x[1] / v,
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
        if time == next_time {
            return x;
        }
        let mut stepper = Dopri5::new(system, time, next_time, 1e-3, x, RTOL, ATOL);
        let _res = stepper.integrate();
        let y = stepper.y_out();
        x = *y.last().unwrap();
        x
    }
}

fn main() -> Result<()> {
    start(Engine::new(Ode {}), "examples/vori/config.toml".to_string())?;
    Ok(())
}
