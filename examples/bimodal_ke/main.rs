#![allow(dead_code)]
#![allow(unused_variables)]
use eyre::Result;
use npcore::prelude::{
    datafile::{CovLine, Infusion, Scenario},
    predict::{Engine, Predict},
    start,
};
use ode_solvers::*;
use std::collections::HashMap;

const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

#[derive(Debug, Clone, Copy)]
struct PopParams {
    ke: f64,
    v: f64,
}
impl PopParams {
    fn new(params: Vec<f64>) -> Self {
        PopParams {
            ke: params[0],
            v: params[1],
        }
    }
}
#[derive(Debug, Clone)]
struct Model<'a> {
    params: PopParams,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    cov: Option<&'a HashMap<String, CovLine>>,
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let ke = self.params.ke;

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

fn eval_outeq(
    params: &PopParams,
    cov: &HashMap<String, CovLine>,
    x: &State,
    time: f64,
    outeq: usize,
) -> f64 {
    match outeq {
        1 => x[0] / params.v,
        _ => panic!("Invalid output equation"),
    }
}

#[derive(Debug, Clone)]
struct Ode {}

impl Predict for Ode {
    fn predict(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let params = PopParams::new(params);
        let mut system = Model {
            params: params,
            _scenario: scenario,
            infusions: vec![],
            cov: None,
        };
        // let scenario = scenario.reorder_with_lag(vec![]);
        let mut yout = vec![];
        let mut x = State::new(0.0);
        let mut index: usize = 0;
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
                    yout.push(eval_outeq(
                        &params,
                        system.cov.unwrap(),
                        &x,
                        event.time,
                        event.outeq.unwrap(),
                    ));
                    // yout.push(x[event.outeq.unwrap() - 1] / params[1]);
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    //TODO: use the last dx as the initial one for the next simulation.
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
    let _result = start(
        Engine::new(Ode {}),
        "examples/bimodal_ke/config.toml".to_string(),
    )?;

    Ok(())
}
