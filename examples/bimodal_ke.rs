use eyre::Result;
use np_core::prelude::*;
use ode_solvers::*;

#[derive(Debug, Clone)]
struct Model<'a> {
    ke: f64,
    _v: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
}
#[derive(Debug, Clone)]
pub struct Infusion {
    pub time: f64,
    pub dur: f64,
    pub amount: f64,
    pub compartment: usize,
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let ke = self.ke;
        // let t = t - self.lag;

        let mut rateiv = [0.0, 0.0];
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] += infusion.amount / infusion.dur;
            }
        }

        ///////////////////// USER DEFINED ///////////////

        dy[0] = -ke * y[0] + rateiv[0];

        //////////////// END USER DEFINED ////////////////
        // for dose in &self.scenario.doses{
        //     if (dose.time + self.lag) > t-(STEP_SIZE/2.) && (dose.time + self.lag) <= t+(STEP_SIZE / 2.) {
        //         y[dose.compartment-1] += dose.dose;
        //     }
        // }
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
        };
        let mut yout = vec![];
        let mut y0 = State::new(0.0);
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
                        y0[event.input.unwrap() - 1] += event.dose.unwrap();
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
        "examples/bimodal_ke.toml".to_string(),
        (0.0, 0.05, 0.0, 0.0),
    )?;

    Ok(())
}
