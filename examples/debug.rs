use std::collections::HashMap;

use eyre::Result;
use ndarray::Array;
use npcore::prelude::{
    datafile::{CovLine, Infusion, Scenario},
    predict::Predict,
};
use ode_solvers::*;

const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

#[derive(Debug, Clone)]
struct Model<'a> {
    ke: f64,
    _v: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    cov: Option<&'a HashMap<String, CovLine>>,
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
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

impl Predict for Ode {
    fn predict(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let mut system = Model {
            ke: params[0],
            _v: params[1],
            _scenario: scenario,
            infusions: vec![],
            cov: None,
        };
        let lag = 0.0;
        let mut yout = vec![];
        let mut x = State::new(0.0);
        let mut index: usize = 0;
        for block in &scenario.blocks {
            //if no code is needed here, remove the blocks from the codebase
            //It seems that blocks is an abstractions we're going to end up not using
            system.cov = Some(&block.covs);
            for event in &block.events {
                let lag_time = event.time + lag;
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        system.infusions.push(Infusion {
                            time: lag_time,
                            dur: event.dur.unwrap(),
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                        // x = simulate_next_state(x, &system, scenario, event, index);
                        if let Some(next_time) = scenario.times.get(index + 1) {
                            if *next_time > event.time {
                                let mut stepper = Dopri5::new(
                                    system.clone(),
                                    event.time,
                                    *next_time,
                                    1e-3,
                                    x,
                                    RTOL,
                                    ATOL,
                                );

                                let _res = stepper.integrate();
                                let y = stepper.y_out();
                                x = *y.last().unwrap();
                            } else if *next_time > event.time {
                                log::error!("Panic: Next event's time is in the past!");
                                panic!("Panic: Next event's time is in the past!");
                            }
                        }
                    } else {
                        //dose
                        if lag > 0.0 {
                            // let mut stepper =
                            //     Rk4::new(system.clone(), event.time, x, lag_time, 0.1);
                            if let Some(next_time) = scenario.times.get(index + 1) {
                                if *next_time < lag_time {
                                    log::error!("Panic: lag time overpasses next observation, not implemented. Stopping.");
                                    panic!("Panic: lag time overpasses next observation, not implemented. Stopping.");
                                }
                                let mut stepper = Dopri5::new(
                                    system.clone(),
                                    event.time,
                                    lag_time,
                                    1e-3,
                                    x,
                                    RTOL,
                                    ATOL,
                                );

                                let _int = stepper.integrate();
                                let y = stepper.y_out();
                                x = *y.last().unwrap();
                            }
                        }

                        x[event.input.unwrap() - 1] += event.dose.unwrap();
                        if let Some(next_time) = scenario.times.get(index + 1) {
                            if *next_time > lag_time {
                                let mut stepper = Dopri5::new(
                                    system.clone(),
                                    lag_time,
                                    *next_time,
                                    1e-3,
                                    x,
                                    RTOL,
                                    ATOL,
                                );

                                let _res = stepper.integrate();
                                let y = stepper.y_out();
                                x = *y.last().unwrap();
                            } else if *next_time > event.time {
                                log::error!("Panic: Next event's time is in the past!");
                                panic!("Panic: Next event's time is in the past!");
                            }
                        }
                    }
                } else if event.evid == 0 {
                    //obs
                    yout.push(x[event.outeq.unwrap() - 1] / params[1]);
                    if let Some(next_time) = scenario.times.get(index + 1) {
                        // let mut stepper = Rk4::new(system.clone(), lag_time, x, *next_time, 0.1);
                        if *next_time > event.time {
                            let mut stepper = Dopri5::new(
                                system.clone(),
                                event.time,
                                *next_time,
                                1e-3,
                                x,
                                RTOL,
                                ATOL,
                            );

                            let _res = stepper.integrate();
                            let y = stepper.y_out();
                            x = *y.last().unwrap();
                        } else if *next_time > event.time {
                            log::error!("Panic: Next event's time is in the past!");
                            panic!("Panic: Next event's time is in the past!");
                        }
                    }
                }
                index += 1;
            }
        }
        yout
    }
}

fn main() -> Result<()> {
    let scenarios =
        npcore::routines::datafile::parse(&"examples/data/bimodal_ke.csv".to_string()).unwrap();
    let scenario = scenarios.first().unwrap();
    // let block = scenario.blocks.get(0).unwrap();
    // dbg!(&block.covs);
    // dbg!(&block.covs.get("WT").unwrap().interp(12.0));

    let sim = Ode {};
    // Vamos a asumir que todos los valores de covs están presentes
    let yobs = Array::from_vec(scenario.obs.clone());
    let ypred =
        Array::from_vec(sim.predict(vec![0.3137412105321884, 116.93967163562775], scenario));
    dbg!(&yobs);
    dbg!(&ypred);
    dbg!(&yobs - &ypred);
    Ok(())
}
