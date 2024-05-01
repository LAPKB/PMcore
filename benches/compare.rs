use pmcore::routines::data::{parse_pmetrics::read_pmetrics, DataTrait};
use pmcore::simulator::analytical::one_compartment_with_absorption;
use pmcore::{prelude::*, simulator::Equation};
use std::path::Path;
type V = nalgebra::DVector<f64>;
fn main() {
    // Run registered benchmarks.
    divan::main();
}
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;
#[derive(Debug, Clone)]
struct Model {
    params: HashMap<String, f64>,
    _scenario: Scenario,
    infusions: Vec<Infusion>,
    cov: Option<HashMap<String, CovLine>>,
}
impl Model {
    pub fn get_param(&self, str: &str) -> f64 {
        *self.params.get(str).unwrap()
    }
}
type State = Vector2<f64>;

// Time uses f64 precision
type Time = f64;
impl ode_solvers::System<Time, State> for Model {
    /// The system function, defining the ordinary differential equations (ODEs) to be solved
    fn system(&self, t: Time, x: &State, dx: &mut State) {
        // Get the parameters from the model
        let ke = self.get_param("ke");
        let ka = self.get_param("ka");

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
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
    }
}

#[derive(Debug, Clone)]
struct Ode {}

impl<'a> Predict<'a> for Ode {
    type Model = Model;
    type State = State;
    fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario) {
        let params = HashMap::from([
            ("ke".to_string(), params[0]),
            ("ka".to_string(), params[1]),
            ("v".to_string(), params[2]),
        ]);
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
            2 => x[1] / v, // Concentration of the central compartment defined by the amount of drug, x[0], divided by the volume of the central compartment, v
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
            // panic!("time error")
            return;
        }
        let mut stepper = Dopri5::new(system.clone(), time, next_time, 1e-3, *x, RTOL, ATOL);
        let _res = stepper.integrate();
        let y = stepper.y_out();
        *x = *y.last().unwrap();
    }
}
const N_EXEC: usize = 10;

#[divan::bench()]
pub fn old_ode() {
    let engine = Engine::new(Ode {});
    let data = parse(&"examples/data/bimodal_ke.csv".to_string()).unwrap();
    let subject = data.first().unwrap();
    for _ in 0..N_EXEC {
        let _ = engine.pred(subject.clone(), vec![0.1, 0.9, 50.0]);
    }
}

#[divan::bench()]
pub fn ode_solvers() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();

    let ode = Equation::new_ode_solvers(
        |x, p, _t, dx, rateiv, _cov| {
            //fetch_cov!(cov, t, creat);
            // fetch_params!(p, ke, ka, _v);
            let ke = p[0];
            let ka = p[1];
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            // fetch_params!(p, _ke, _ka, v);
            let v = p[2];
            V::from_vec(vec![x[0] / v, x[1] / v])
        },
    );

    for _ in 0..N_EXEC {
        let _ = ode.simulate_subject(first_subject, &vec![0.1, 0.9, 50.0]);
    }
}

#[divan::bench()]
pub fn ode() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();

    let ode = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            //fetch_cov!(cov, t, creat);
            // fetch_params!(p, ke, ka, _v);
            let ke = p[0];
            let ka = p[1];
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            // fetch_params!(p, _ke, _ka, v);
            let v = p[2];
            V::from_vec(vec![x[0] / v, x[1] / v])
        },
    );

    for _ in 0..N_EXEC {
        let _ = ode.simulate_subject(first_subject, &vec![0.1, 0.9, 50.0]);
    }
}

#[divan::bench()]
pub fn analytical() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();

    let analytical = Equation::new_analytical(
        one_compartment_with_absorption,
        |_p, _cov| {},
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            // fetch_params!(p, _ke, _ka, v);
            let v = p[2];
            V::from_vec(vec![x[0] / v, x[1] / v])
        },
    );
    for _ in 0..N_EXEC {
        let _ = analytical.simulate_subject(first_subject, &vec![0.1, 0.9, 50.0]);
    }
}
