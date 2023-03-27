use eyre::Result;
use np_core::prelude::{datafile::get_cov, *};
use ode_solvers::*;

const STEP_SIZE: f64 = 0.1;

struct Model<'a> {
    ka: f64,
    ke: f64,
    _v: f64,
    lag: f64,
    scenario: &'a Scenario,
    wt: f64
}

type State = Vector2<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, y: &mut State, dy: &mut State) {
        let ka = self.ka;
        let ke = self.ke;
        ///////////////////// USER DEFINED ///////////////
        dy[0] = -ka * y[0];
        dy[1] = ka * y[0] - ke * y[1];
        //////////////// END USER DEFINED ////////////////
        for dose in &self.scenario.doses {
            if (dose.time + self.lag) > t - (STEP_SIZE / 2.)
                && (dose.time + self.lag) <= t + (STEP_SIZE / 2.)
            {
                y[dose.compartment] += dose.dose;
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
         // let wt = get_cov(&self.scenario.covariates, "WT".to_string()).unwrap().values.first().unwrap();
        // let t = t - self.lag;
        let system = Model {
            ka: params[0],
            ke: params[1],
            _v: params[2],
            lag: params[3],
            scenario,
            wt: *get_cov(&scenario.covariates, "WT".to_string()).unwrap().values.first().unwrap()
        };
        let y0 = State::new(0.0, 0.0);
        let mut stepper = Rk4::new(system, tspan[0], y0, tspan[1], STEP_SIZE);
        // let mut stepper = Dopri5::new(system, 0.0, 20.0, 1.0e-5, y0, 1.0e-14, 1.0e-14);
        let _res = stepper.integrate();
        let x = stepper.x_out().to_vec();
        let y = stepper.y_out();
        let mut yout: Vec<Vec<f64>> = vec![];
        let v = params[2];
        ///////////////////// ONE PER OUTPUT EQUATION ///////////////
        let y0: Vec<f64> = y
            .iter()
            .map(|y| {
                ///////////////////// USER DEFINED ///////////////
                y[1] / v
                //////////////// END USER DEFINED ////////////////
            })
            .collect();
        yout.push(y0);
        //////////////// END ONE PER OUTPUT EQUATION ////////////////
        (x, yout)
    }
}

fn main() -> Result<()> {
    start(
        Engine::new(Sim {}),
        "examples/two_eq_lag.toml".to_string(),
        (0.1, 0.25, -0.001, 0.0),
    )?;
    Ok(())
}
