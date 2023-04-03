use eyre::Result;
use np_core::prelude::*;
use ode_solvers::*;

const STEP_SIZE: f64 = 1.0 / 60.0; // one step per minute

#[derive(Debug, Clone)]
struct Model<'a> {
    ke: f64,
    _v: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    dose: Option<Dose>,
}
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
                        system.dose = Some(Dose {
                            time: event.time,
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    }
                }
                // let mut stepper = Dopri5::new(system.clone(),time,event.time,0.001,y0,1.0e-14,1.0e-14,);

                let mut stepper = Rk4::new(system.clone(), time, y0, event.time, STEP_SIZE * 6.0);
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
    // let scenarios = np_core::base::datafile::parse(&"examples/bimodal_ke.csv".to_string()).unwrap();
    // let scenario = scenarios.first().unwrap();
    start(
        Engine::new(Sim {}),
        "examples/bimodal_ke/config.toml".to_string(),
        (0.0, 0.05, 0.0, 0.0),
    )?;
    // let sim = Sim {};

    // // dbg!(&scenario);
    // dbg!(&scenario.obs);
    // dbg!(sim.simulate(vec![0.3142161965370178, 119.59214568138123], scenario));

    Ok(())
}
