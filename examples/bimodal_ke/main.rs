use pmcore::prelude::*;

// Constants for the absolute and relative tolerance for the dynamic steps used for solving the ODEs
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

// Define the state vector, which must be equal to the number of compartments in the model
// These are re-exported from the `nalgebra`-crate by `ode_solvers`, see https://github.com/srenevey/ode-solvers?tab=readme-ov-file#type-alias-definition
// In brief, for up to 6 compartments, use VectorN<f64>, N being the number of compartments.
// For more than 6 compartments, use `nalgebra::SVector<f64, N>`, where N is the number of compartments.
type State = Vector1<f64>;

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
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        // Get the parameters from the model
        let ke = self.get_param("ke");

        // Get the infusions that are active at time `t`
        let mut rateiv = [0.0];
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] += infusion.amount / infusion.dur;
            }
        }
        // The ordinary differential equations (ODEs) are defined here
        // This example is a one-compartmental model with first-order elimination, and intravenous infusions

        ////// ODE //////
        dy[0] = -ke * y[0] + rateiv[0];
    }
}

#[derive(Debug, Clone)]
struct Ode {}

impl<'a> Predict<'a> for Ode {
    type Model = Model;
    type State = State;
    fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario) {
        let params = HashMap::from([("ke".to_string(), params[0]), ("v".to_string(), params[1])]);
        (
            Model {
                params,
                _scenario: scenario.clone(), //TODO remove
                infusions: vec![],
                cov: None,
            },
            scenario.reorder_with_lag(vec![(0.0, 1)]),
        )
    }

    // This function is used to get the output from the model, defined by the output equations (outeq) supplied by the user
    fn get_output(&self, time: f64, x: &Self::State, system: &Self::Model, outeq: usize) -> f64 {
        // Get parameters from the model also used for calculating the output equations
        let v = system.get_param("v");
        #[allow(unused_variables)]
        let t = time;
        match outeq {
            1 => x[0] / v, // Concentration of the central compartment defined by the amount of drug, x[0], divided by the volume of the central compartment, v
            _ => panic!("Invalid output equation"),
        }
    }

    // Set the initial state of the compartments
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
        if time >= next_time {
            panic!("time error")
        }
        let mut stepper = Dopri5::new(system.clone(), time, next_time, 1e-3, *x, RTOL, ATOL);
        let _res = stepper.integrate();
        let y = stepper.y_out();
        *x = *y.last().unwrap();
    }
}

fn main() -> Result<()> {
    unimplemented!();
    // // Main entrypoint, see `entrypoints.rs` for more details
    // let _result = start(
    //     Engine::new(Ode {}),
    //     "examples/bimodal_ke/config.toml".to_string(),
    // )?;

    // Ok(())
}
