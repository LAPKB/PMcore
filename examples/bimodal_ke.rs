use eyre::Result;
use np_core::prelude::*;
use ode_solvers::*;

struct Model<'a> {
    ke: f64,
    _v: f64,
    scenario: &'a Scenario,
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, y: &mut State, dy: &mut State) {
        let ke = self.ke;
        // let t = t - self.lag;

        let mut rateiv = [0.0, 0.0];
        for infusion in &self.scenario.infusions {
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

struct Sim {}

impl Simulate for Sim {
    fn simulate(&self, params: Vec<f64>, scenario: &Scenario) -> (Vec<f64>, Vec<Vec<f64>>) {
        let system = Model {
            ke: params[0],
            _v: params[1],
            scenario,
        };
        // [
        //     *scenario.time.first().unwrap(),
        //     *scenario.time.last().unwrap(),
        // ],
        let y0 = State::new(0.0);
        let mut stepper = Rk4::new(system, 0.0, y0, 1.0, 0.1);
        let _res = stepper.integrate();
        let x = stepper.x_out().to_vec();
        let y = stepper.y_out();

        let mut yout: Vec<Vec<f64>> = vec![];
        let y0: Vec<f64> = y.iter().map(|y| y[0] / params[1]).collect();
        yout.push(y0);

        (x, yout)
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
