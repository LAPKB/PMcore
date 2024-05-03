#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use ode_solvers::dop853::*;
use pmcore::prelude::*;

// Constants for the absolute and relative tolerance for the dynamic steps used for solving the ODEs
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

// Define the state vector, which must be equal to the number of compartments in the model
// These are re-exported from the `nalgebra`-crate by `ode_solvers`, see https://github.com/srenevey/ode-solvers?tab=readme-ov-file#type-alias-definition
// In brief, for up to 6 compartments, use VectorN<f64>, N being the number of compartments.
// For more than 6 compartments, use `nalgebra::SVector<f64, N>`, where N is the number of compartments.
type State = SVector<f64, 2>;
// Time uses f64 precision
type Time = f64;

// This is the main structure for the model
// It holds the parameters (defined in config.toml), the scenarios (i.e. datafile), the (possible) infusions and the covariates if any
#[derive(Debug, Clone)]
struct Model {
    params: HashMap<String, f64>,
    _scenario: Scenario,
    infusions: Vec<Infusion>,
    cov: Option<HashMap<String, CovLine>>,
}
// This is a helper function to get the parameter value by name
impl Model {
    pub fn get_param(&self, str: &str) -> f64 {
        *self.params.get(str).unwrap()
    }
}

impl ode_solvers::System<Time, State> for Model {
    /// The system function, defining the ordinary differential equations (ODEs) to be solved
    fn system(&self, t: Time, x: &State, dx: &mut State) {
        // Get the parameters, covariates and secondary equations from the model
        let cls = self.get_param("cls");
        let fm = self.get_param("fm");
        let k20 = self.get_param("k20");
        let relv = self.get_param("relv");
        let theta1 = self.get_param("theta1");
        let theta2 = self.get_param("theta2");
        let vs = self.get_param("vs");
        let cov = self.cov.as_ref().unwrap();
        let age = cov.get("age").unwrap().interp(t);
        let wt = cov.get("wt").unwrap().interp(t);
        let ht = cov.get("ht").unwrap().interp(t);
        let bsa = cov.get("bsa").unwrap().interp(t);
        let pkvisit = cov.get("pkvisit").unwrap().interp(t);
        let bmi = cov.get("bmi").unwrap().interp(t);
        let bmip = cov.get("bmip").unwrap().interp(t);
        let dxa_bf = cov.get("dxa_bf").unwrap().interp(t);
        let block = cov.get("block").unwrap().interp(t);

        let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
        let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
        let ke = cl / v;
        let v2 = relv * v;

        let mut rateiv = [0.0]; //TODO: hardcoded
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] = infusion.amount / infusion.dur;
            }
        }
        dx[0] = rateiv[0] - ke * x[0] * (1.0 - fm) - fm * x[0];
        dx[1] = fm * x[0] - k20 * x[1];
    }
}

#[derive(Debug, Clone)]
struct Ode {}

impl<'a> Predict<'a> for Ode {
    type Model = Model;
    type State = State;
    // This function is used initialize the system by setting parameter names and initial empty structs.
    fn initial_system(&self, parameters: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario) {
        let mut params = HashMap::new();
        // params.insert("ke".to_string(), params[0].clone());
        // params.insert("v".to_string(), params[1].clone());
        params.insert("cls".to_string(), parameters[0].clone());
        params.insert("fm".to_string(), parameters[1].clone());
        params.insert("k20".to_string(), parameters[2].clone());
        params.insert("relv".to_string(), parameters[3].clone());
        params.insert("theta1".to_string(), parameters[4].clone());
        params.insert("theta2".to_string(), parameters[5].clone());
        params.insert("vs".to_string(), parameters[6].clone());
        let system = Model {
            params,
            _scenario: scenario.clone(), //TODO remove
            infusions: vec![],
            cov: None,
        };
        let cls = system.get_param("cls");
        let fm = system.get_param("fm");
        let k20 = system.get_param("k20");
        let relv = system.get_param("relv");
        let theta1 = system.get_param("theta1");
        let theta2 = system.get_param("theta2");
        let vs = system.get_param("vs");
        (
            system, // scenario.reorder_with_lag(vec![(0.0, 1)]))
            scenario,
        )
    }
    // This function is used to get the output from the model, defined by the output equations (outeq) supplied by the user
    fn get_output(&self, t: f64, x: &Self::State, system: &Self::Model, outeq: usize) -> f64 {
        // let v = system.get_param("v");
        // match outeq {
        //     1 => x[0] / v,
        //     _ => panic!("Invalid output equation"),
        // }
        let cls = system.get_param("cls");
        let fm = system.get_param("fm");
        let k20 = system.get_param("k20");
        let relv = system.get_param("relv");
        let theta1 = system.get_param("theta1");
        let theta2 = system.get_param("theta2");
        let vs = system.get_param("vs");
        let cov = system.cov.as_ref().unwrap();
        let age = cov.get("age").unwrap().interp(t);
        let wt = cov.get("wt").unwrap().interp(t);
        let ht = cov.get("ht").unwrap().interp(t);
        let bsa = cov.get("bsa").unwrap().interp(t);
        let pkvisit = cov.get("pkvisit").unwrap().interp(t);
        let bmi = cov.get("bmi").unwrap().interp(t);
        let bmip = cov.get("bmip").unwrap().interp(t);
        let dxa_bf = cov.get("dxa_bf").unwrap().interp(t);
        let block = cov.get("block").unwrap().interp(t);

        let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
        let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
        let ke = cl / v;
        let v2 = relv * v;

        match outeq {
            1 => x[0] / v,
            2 => x[1] / v2,
            _ => panic!("Invalid output equation"),
        }
    }
    // This function is used to initialize the compartments
    //TODO: Handle non-zero initial conditions
    fn initial_state(&self) -> State {
        State::default()
    }
    // Add any possible infusions
    fn add_infusion(&self, system: &mut Self::Model, infusion: Infusion) {
        system.infusions.push(infusion);
    }
    // Add any possible covariates
    fn add_covs(&self, system: &mut Self::Model, cov: Option<HashMap<String, CovLine>>) {
        system.cov = cov;
    }
    // Add any possible doses
    fn add_dose(&self, state: &mut Self::State, dose: f64, compartment: usize) {
        state[compartment] += dose;
    }
    // Perform a "step" of the model, i.e. solve the ODEs from the current time to the next time
    // In the next step, we use this result as the initial state
    fn state_step(&self, x: &mut Self::State, system: &Self::Model, time: f64, next_time: f64) {
        if time > next_time {
            panic!("time error")
        } else if time == next_time {
            return;
        }
        let mut stepper = Dop853::new(system.clone(), time, next_time, 1e-3, *x, RTOL, ATOL);
        let res = stepper.integrate();
        let y = stepper.y_out();

        *x = *y.last().unwrap();
    }
}

fn main() -> Result<()> {
    unimplemented!();
    // start(Engine::new(Ode {}), "examples/meta/config.toml".to_string())?;
    // Ok(())
}
