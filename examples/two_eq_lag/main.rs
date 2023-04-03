use eyre::Result;
use np_core::prelude::*;
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
    fn system(&mut self, t: Time, y: &State, dy: &mut State) {
        let ka = self.ka;
        let ke = self.ke;
        let lag = self.lag;
        ///////////////////// USER DEFINED ///////////////
        dy[0] = -ka * y[0];
        dy[1] = ka * y[0] - ke * y[1];
        //////////////// END USER DEFINED ////////////////

        if let Some(dose) = &self.dose {
            if t >= dose.time + lag {
                dy[dose.compartment] += dose.amount;
                self.dose = None;
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
        let mut yout = vec![];
        let mut y0 = State::new(0.0, 0.0);
        let mut time = 0.0;
        for block in &scenario.blocks {
            for event in block {
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
                        system.dose = Some(Dose {
                            time: event.time,
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    }
                }
                // let mut stepper = Dopri5::new(
                //     system.clone(),
                //     time,
                //     event.time,
                //     0.001,
                //     y0,
                //     1.0e-14,
                //     1.0e-14,
                // );
                let mut stepper = Rk4::new(system.clone(), time, y0, event.time, 0.1);
                let _res = stepper.integrate();
                let y = stepper.y_out();
                y0 = match y.last() {
                    Some(y) => *y,
                    None => y0,
                };
                if event.evid == 0 {
                    //obs
                    yout.push(y0[event.outeq.unwrap() - 1] / params[1]);
                }
                time = event.time;
            }
        }
        yout
    }
}

fn main() -> Result<()> {
    start(
        Engine::new(Sim {}),
        "examples/two_eq_lag/config.toml".to_string(),
        (0.1, 0.25, -0.001, 0.0),
    )?;
    Ok(())
}
