use eyre::Result;
use np_core::prelude::{datafile::Event, *};
use ode_solvers::*;

#[derive(Debug, Clone)]
struct Model<'a> {
    ka: f64,
    ke: f64,
    _v: f64,
    lag: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    dose: Option<Dose>,
}

type State = Vector2<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&mut self, t: Time, y: &mut State, dy: &mut State) {
        let ka = self.ka;
        let ke = self.ke;
        ///////////////////// USER DEFINED ///////////////
        dy[0] = -ka * y[0];
        dy[1] = ka * y[0] - ke * y[1];
        //////////////// END USER DEFINED ////////////////

        if let Some(dose) = &self.dose {
            if dose.time > t - (0.1 / 2.) && dose.time <= t + (0.1 / 2.) {
                y[dose.compartment] += dose.amount;
            }
        }
    }
}

struct Sim {}

#[derive(Debug, Clone)]
pub struct Infusion {
    pub time: f64,
    pub dur: f64,
    pub amount: f64,
    pub compartment: usize,
}
#[derive(Debug, Clone)]
pub struct Dose {
    pub time: f64,
    pub amount: f64,
    pub compartment: usize,
}

impl Simulate for Sim {
    fn simulate(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let mut system = Model {
            ka: params[0],
            ke: params[1],
            _v: params[2],
            lag: params[3],
            _scenario: scenario,
            infusions: vec![],
            dose: None,
        };
        let lag = system.lag; // or 0.0
        let mut yout = vec![];
        let mut y0 = State::new(0.0, 0.0);

        for (block_index, block) in scenario.blocks.iter().enumerate() {
            for (event_index, event) in block.iter().enumerate() {
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        system.infusions.push(Infusion {
                            time: event.time + lag,
                            dur: event.dur.unwrap(),
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    } else {
                        //dose
                        system.dose = Some(Dose {
                            time: event.time + lag,
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    }
                } else if event.evid == 0 {
                    //obs
                    //TODO: implement outeq
                    yout.push(y0[1] / params[2]);
                }
                if let Some(next_event) = next_event(&scenario, block_index, event_index) {
                    let mut stepper =
                        Rk4::new(system.clone(), event.time, y0, next_event.time, 0.1);
                    let _res = stepper.integrate();
                    let y = stepper.y_out();
                    dbg!(&y);
                    y0 = match y.last() {
                        Some(y) => *y,
                        None => y0,
                    };
                    dbg!((event.time, next_event.time, y0 / params[2]));
                }
            }
        }
        yout
    }
}

fn next_event(scenario: &Scenario, block_index: usize, event_index: usize) -> Option<Event> {
    let mut next_event = None;
    if let Some(event) = scenario.blocks[block_index].get(event_index + 1) {
        next_event = Some(event.clone());
    } else if let Some(block) = scenario.blocks.get(block_index + 1) {
        if let Some(event) = block.first() {
            next_event = Some(event.clone());
        }
    }
    next_event
}

fn main() -> Result<()> {
    let scenarios =
        np_core::base::datafile::parse(&"examples/data/two_eq_lag.csv".to_string()).unwrap();
    let scenario = scenarios.first().unwrap();
    let sim = Sim {};

    // dbg!(&scenario);

    dbg!(sim.simulate(
        vec![
            0.12197456836700439,
            0.08711592574119567,
            31.96331477165222,
            3.9866789817810058
        ],
        scenario
    ));
    dbg!(&scenario.obs);

    Ok(())
}
