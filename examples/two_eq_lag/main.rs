use eyre::Result;
use npcore::prelude::{
    datafile::{CovLine, Infusion},
    *,
};
use ode_solvers::*;
use std::collections::HashMap;
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;
#[derive(Debug, Clone)]
struct Model<'a> {
    ka: f64,
    ke: f64,
    lag: f64,
    _v: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    cov: Option<&'a HashMap<String, CovLine>>,
}

type State = SVector<f64, 2>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, x: &State, dx: &mut State) {
        // Random parameters
        let ka = self.ka;
        let ke = self.ke;
        // let lag = self.lag;
        // Covariates
        let _wt = self.cov.unwrap().get("WT").unwrap().interp(t);
        let mut rateiv = [0.0];
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] += infusion.amount / infusion.dur;
            }
        }
        ///////////////////// USER DEFINED ///////////////
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1];
        //////////////// END USER DEFINED ////////////////
    }
}

struct Ode {}

impl Predict for Ode {
    fn predict(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let mut system = Model {
            ka: params[0],
            ke: params[1],
            lag: params[2],
            _v: params[3],
            _scenario: scenario,
            infusions: vec![],
            cov: None,
        };
        let scenario = scenario.reorder_with_lag(vec![(system.lag, 1)]);
        let mut yout = vec![];
        let mut x = State::new(0.0, 0.0);
        let mut index = 0;
        for block in &scenario.blocks {
            system.cov = Some(&block.covs);
            for event in &block.events {
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        system.infusions.push(Infusion {
                            time: event.time,
                            dur: event.dur.unwrap(),
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    } else {
                        //dose
                        x[event.input.unwrap() - 1] += event.dose.unwrap();
                    }
                } else if event.evid == 0 {
                    //obs
                    yout.push(x[1] / params[3]);
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    // let mut stepper = Rk4::new(system.clone(), lag_time, x, *next_time, 0.1);
                    let mut stepper =
                        Dopri5::new(system.clone(), event.time, *next_time, 1e-3, x, RTOL, ATOL);
                    let _res = stepper.integrate();
                    let y = stepper.y_out();
                    x = *y.last().unwrap();
                }
                index += 1;
            }
        }
        yout
    }
}

fn main() -> Result<()> {
    // let scenarios = parse(&"examples/data/two_eq_lag.csv".to_string())
    //     .ok()
    //     .unwrap();
    // let scenario = scenarios.first().unwrap();
    // let ode = Ode {};
    // let params = vec![
    //     0.10007869720458984,
    //     0.0999935963869095,
    //     0.6458234786987305,
    //     119.99048137664795,
    // ];
    // let y = ode.predict(params, scenario);
    // println!("{:?}", y);
    // println!("{:?}", scenario.obs);
    start(
        Engine::new(Ode {}),
        "examples/two_eq_lag/config.toml".to_string(),
    )?;

    // simulate(
    //     Engine::new(Ode {}),
    //     "examples/two_eq_lag/sim_config.toml".to_string(),
    // )?;

    Ok(())
}
