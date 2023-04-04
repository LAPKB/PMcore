use eyre::Result;
use np_core::prelude::{
    datafile::{Dose, Event, Infusion},
    *,
};
use ode_solvers::*;

#[derive(Debug, Clone)]
struct Model<'a> {
    ke: f64,
    _v: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    dose: Option<Dose>,
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&mut self, t: Time, y: &mut State, dy: &mut State) {
        let ke = self.ke;

        let lag = 0.0;

        let mut rateiv = [0.0];
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] = infusion.amount / infusion.dur;
            }
        }

        ///////////////////// USER DEFINED ///////////////

        dy[0] = -ke * y[0] + rateiv[0];

        //////////////// END USER DEFINED ////////////////

        if let Some(dose) = &self.dose {
            if t >= dose.time + lag {
                dy[dose.compartment] += dose.amount;
                self.dose = None;
            }
        }
    }
}
#[derive(Debug, Clone)]
struct Sim {}

impl Simulate for Sim {
    fn simulate(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let mut system = Model {
            ke: params[0],
            _v: params[1],
            _scenario: scenario,
            infusions: vec![],
            dose: None,
        };
        let lag = 0.0;
        let mut yout = vec![];
        let mut y0 = State::new(0.0);
        let mut index: usize = 0;
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
                    yout.push(y0[event.outeq.unwrap() - 1] / params[1]);
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    if next_event(&scenario, block_index, event_index)
                        .unwrap()
                        .time
                        != *next_time
                    {
                        //stop exectuion
                        dbg!(
                            next_event(&scenario, block_index, event_index)
                                .unwrap()
                                .time,
                            *next_time
                        );
                        panic!("error");
                    }
                    let mut stepper = Rk4::new(system.clone(), event.time, y0, *next_time, 0.1);
                    let _res = stepper.integrate();
                    let y = stepper.y_out();
                    y0 = *y.last().unwrap();
                }
                index += 1;
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
        np_core::base::datafile::parse(&"examples/data/bimodal_ke.csv".to_string()).unwrap();
    let scenario = scenarios.get(2).unwrap();
    // let sim = Sim {};
    let block = scenario.blocks.first().unwrap();
    // let first = block.first().unwrap();
    // let second = block.get(1).unwrap();
    // dbg!(first);
    // dbg!(second);
    dbg!(block.len());

    // // dbg!(&scenario);
    // dbg!(&scenario.obs);
    // dbg!(sim.simulate(vec![0.3142161965370178, 119.59214568138123], scenario));

    Ok(())
}
