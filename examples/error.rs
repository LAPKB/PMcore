#![allow(dead_code)]
#![allow(unused_variables)]

use eyre::Result;
use np_core::prelude::*;
use ode_solvers::*;

const STEP_SIZE: f64 = 0.1;

struct Model<'a> {
    ka: f64,
    ke: f64,
    v: f64,
    lag: f64,
    scenario: &'a Scenario,
}

type State = Vector2<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, x: &mut State, dx: &mut State) {
        let ka = self.ka;
        let ke = self.ke;
        let v = self.v;
        // let lag = self.lag;
        let mut rateiv = [0.0, 0.0];
        for infusion in &self.scenario.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] += infusion.amount / infusion.dur;
            }
        }
        let t = t - self.lag;
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1];

        for dose in &self.scenario.doses {
            if (dose.time + self.lag) > t-(STEP_SIZE/2.) && (dose.time + self.lag) <= t+(STEP_SIZE / 2.) {
                x[dose.compartment] += dose.dose;
            }
        }
    }
}
struct Sim {}
impl Simulate for Sim {
    fn simulate(
        &self,
        params: Vec<f64>,
        tspan: [f64; 2],
        scenario: &Scenario,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let system = Model {
            ka: params[0],
            ke: params[1],
            v: params[2],
            lag: params[3],
            scenario,
        };
        let y0 = State::new(0.0, 0.0);
        let mut stepper = Rk4::new(system, tspan[0], y0, tspan[1], STEP_SIZE);
        let _res = stepper.integrate();
        let x = stepper.x_out().to_vec();
        let y = stepper.y_out();
        let mut yout: Vec<Vec<f64>> = vec![];
        let ka = params[0];
        let ke = params[1];
        let v = params[2];
        let lag = params[3];
        let y0: Vec<f64> = y.iter().map(|x| x[1] / v).collect();
        yout.push(y0);
        (x, yout)
    }
}

fn main() -> Result<()> {
    start(
        Engine::new(Sim {}),
        vec![
            (0.100000, 0.900000),
            (0.001000, 0.100000),
            (30.000000, 120.000000),
            (0.000000, 4.000000),
        ],
        "examples/error.toml".to_string(),
        (0.100000, 0.250000, -0.001000, 0.000000),
    )?;
    Ok(())
}
