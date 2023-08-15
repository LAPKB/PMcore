#![allow(dead_code)]
#![allow(unused_variables)]
use eyre::Result;
use np_core::prelude::{
    datafile::{CovLine, Infusion},
    *,
};
use ode_solvers::*;
use std::collections::HashMap;
#[derive(Debug, Clone)]
struct Model<'a> {
    age50: f64,
    fa1: f64,
    hill: f64,
    ka: f64,
    kcp: f64,
    km: f64,
    kpc: f64,
    vc0: f64,
    vmax0: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    cov: Option<&'a HashMap<String, CovLine>>,
}

type State = SVector<f64, 3>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, x: &State, dx: &mut State) {
        let age50 = self.age50;
        let fa1 = self.fa1;
        let hill = self.hill;
        let ka = self.ka;
        let kcp = self.kcp;
        let km = self.km;
        let kpc = self.kpc;
        let vc0 = self.vc0;
        let vmax0 = self.vmax0;
        let wt = self.cov.unwrap().get("wt").unwrap().interp(t);
        let ast = self.cov.unwrap().get("ast").unwrap().interp(t);
        let alt = self.cov.unwrap().get("alt").unwrap().interp(t);
        let age = self.cov.unwrap().get("age").unwrap().interp(t);

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

impl Predict for Ode {
    fn predict(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let mut system = Model {
            age50: params[0],
            fa1: params[1],
            hill: params[2],
            ka: params[3],
            kcp: params[4],
            km: params[5],
            kpc: params[6],
            vc0: params[7],
            vmax0: params[8],
            _scenario: scenario,
            infusions: vec![],
            cov: None,
        };
        let lag = 0.0;
        let mut yout = vec![];
        let mut x = State::new(0.0, 0.0, 0.0);
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
                                    0.01,
                                    x,
                                    1e-4,
                                    1e-4,
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
                            let mut stepper = Dopri5::new(
                                system.clone(),
                                event.time,
                                lag_time,
                                0.01,
                                x,
                                1e-4,
                                1e-4,
                            );

                            let _int = stepper.integrate();
                            let y = stepper.y_out();
                            x = *y.last().unwrap();
                        }

                        x[event.input.unwrap() - 1] += event.dose.unwrap();
                        if let Some(next_time) = scenario.times.get(index + 1) {
                            if *next_time > event.time {
                                let mut stepper = Dopri5::new(
                                    system.clone(),
                                    lag_time,
                                    *next_time,
                                    0.01,
                                    x,
                                    1e-4,
                                    1e-4,
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
                    yout.push(eval_outeq(
                        &params,
                        system.cov.unwrap(),
                        &x,
                        event.time,
                        event.outeq.unwrap(),
                    ));
                    if let Some(next_time) = scenario.times.get(index + 1) {
                        // let mut stepper = Rk4::new(system.clone(), lag_time, x, *next_time, 0.1);
                        if *next_time > event.time {
                            let mut stepper = Dopri5::new(
                                system.clone(),
                                event.time,
                                *next_time,
                                0.01,
                                x,
                                1e-4,
                                1e-4,
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

fn eval_outeq(
    params: &Vec<f64>,
    cov: &HashMap<String, CovLine>,
    x: &State,
    time: f64,
    outeq: usize,
) -> f64 {
    let age50 = params[0];
    let fa1 = params[1];
    let hill = params[2];
    let ka = params[3];
    let kcp = params[4];
    let km = params[5];
    let kpc = params[6];
    let vc0 = params[7];
    let vmax0 = params[8];
    let wt = cov.get("wt").unwrap().interp(time);
    let ast = cov.get("ast").unwrap().interp(time);
    let alt = cov.get("alt").unwrap().interp(time);
    let age = cov.get("age").unwrap().interp(time);
    let vm = vmax0 * wt.powf(0.75);
    let v = vc0 * wt;
    let fage = age.powf(hill) / (age50.powf(hill) + age.powf(hill));
    match outeq {
        1 => x[1] / v,
        _ => panic!("Invalid output equation"),
    }
}

fn main() -> Result<()> {
    start(Engine::new(Ode {}), "examples/vori/config.toml".to_string())?;
    Ok(())
}
