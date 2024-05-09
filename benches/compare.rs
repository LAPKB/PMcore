use pmcore::prelude::data::{pmetrics::read_pmetrics, DataTrait};
use pmcore::simulator::analytical::one_compartment_with_absorption;
use pmcore::{prelude::*, simulator::Equation};
use std::path::Path;

// type V = nalgebra::DVector<f64>;
// type V = nalgebra::SVector<f64, 3>;

const PATH: &str = "examples/data/two_eq_lag.csv";
const SPP: [f64; 4] = [
    0.022712449789047243, //ke
    0.48245882034301757,  //ka
    71.28352475166321,    //v
    // 0.5903420448303222,   //tlag
    0.0,
];

use diol::prelude::*;
/// baseline uses the old simulator + the old sde solver No dynamic dispatching
/// ode_solvers_* use the old ode solver
/// ode_* use the new ode solver
/// _os uses the old simulator
/// _ns uses the new simulator
fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            // baseline,
            analytical_ns,
            // analytical_os,
            // ode_solvers_os,
            diffsol_ns,
            // diffsol_os,
        ],
        [4, 8, 16, 128],
    );
    bench.run()?;
    Ok(())
}

// pub fn baseline(bencher: Bencher, len: usize) {
//     let engine = Engine::new(Ode {});
//     let data = parse(&PATH.to_string()).unwrap();
//     let scenario = data.first().unwrap();
//     let scenario = &scenario.reorder_with_lag(vec![(SPP[3], 1)]);

//     bencher.bench(|| {
//         for _ in 0..len {
//             black_box(engine.pred(scenario.clone(), SPP.to_vec()));
//         }
//     });
// }
pub fn analytical_ns(bencher: Bencher, len: usize) {
    let data = read_pmetrics(Path::new(PATH)).unwrap();
    let subjects = data.get_subjects();
    let first_subject = subjects.first().unwrap();

    let analytical = Equation::new_analytical(
        one_compartment_with_absorption,
        |_p, _cov| {},
        |p| {
            fetch_params!(p, _ke, _ka, _v, tlag);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _ka, v, _tlag);
            y[0] = x[0] / v;
            y[1] = x[1] / v;
        },
        (4, 2),
    );
    bencher.bench(|| {
        for _ in 0..len {
            black_box(analytical.simulate_subject(&first_subject, &SPP.to_vec()));
        }
    });
}

pub fn diffsol_ns(bencher: Bencher, len: usize) {
    let data = read_pmetrics(Path::new(PATH)).unwrap();
    let subjects = data.get_subjects();
    let first_subject = subjects.first().unwrap();

    let ode = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, ka, _v, _tlag);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |p| {
            fetch_params!(p, _ke, _ka, _v, tlag);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _ka, v, _tlag);
            y[0] = x[0] / v;
            y[1] = x[1] / v;
        },
        (4, 2),
    );
    bencher.bench(|| {
        for _ in 0..len {
            black_box(ode.simulate_subject(&first_subject, &SPP.to_vec()));
        }
    });
}
// pub fn diffsol_os(bencher: Bencher, len: usize) {
//     let data = parse(&PATH.to_string()).unwrap();
//     let scenario = data.first().unwrap();
//     let scenario = &scenario.reorder_with_lag(vec![(SPP[3], 1)]);

//     let ode = Equation::new_ode(
//         |x, p, _t, dx, rateiv, _cov| {
//             fetch_params!(p, ke, ka, _v, _tlag);
//             dx[0] = -ka * x[0];
//             dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
//         },
//         |p| {
//             fetch_params!(p, _ke, _ka, _v, tlag);
//             lag! {0=>tlag}
//         },
//         |_p| fa! {},
//         |_p, _t, _cov, _x| {},
//         |x, p, _t, _cov, y| {
//             fetch_params!(p, _ke, _ka, v, _tlag);
//             y[0] = x[0] / v;
//             y[1] = x[1] / v;
//         },
//         (4, 2),
//     );
//     bencher.bench(|| {
//         for _ in 0..len {
//             black_box(ode.simulate_scenario(scenario, &SPP.to_vec()));
//         }
//     });
// }

// const ATOL: f64 = 1e-4;
// const RTOL: f64 = 1e-4;
// #[derive(Debug, Clone)]
// struct Model {
//     params: Vec<f64>,
//     _scenario: Scenario,
//     infusions: Vec<Infusion>,
//     cov: Option<HashMap<String, CovLine>>,
//     // diffeq: DiffEq,
// }

// impl ode_solvers::System<Time, State> for Model {
//     /// The system function, defining the ordinary differential equations (ODEs) to be solved
//     fn system(&self, t: Time, x: &State, dx: &mut State) {
//         // Get the parameters from the model

//         // Get the infusions that are active at time `t`
//         let mut rateiv = vec![0.0];
//         for infusion in &self.infusions {
//             if t >= infusion.time && t <= (infusion.dur + infusion.time) {
//                 rateiv[infusion.compartment] += infusion.amount / infusion.dur;
//             }
//         }
//         // The ordinary differential equations (ODEs) are defined here
//         // This example is a one-compartmental model with first-order elimination, and intravenous infusions

//         ////// ODE //////
//         // (self.diffeq)(
//         //     x,
//         //     &V::from_vec(self.params.clone()),
//         //     t,
//         //     dx,
//         //     V::from_vec(rateiv),
//         //     &Covariates::new(),
//         // )
//         let ke = self.params[0];
//         let ka = self.params[1];
//         dx[0] = -ka * x[0];
//         dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
//     }
// }

// // #[derive(Debug, Clone)]
// // struct Ode {
// //     // diffeq: DiffEq,
// // }
// // type State = SVector<f64, 2>;
// // // type State = DVector<f64>;
// // type Time = f64;
// // impl<'a> Predict<'a> for Ode {
// //     type Model = Model;
// //     type State = State;
// //     fn initial_state(&self) -> State {
// //         SVector::default()
// //         // DVector::zeros(2)
// //     }
// //     fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario) {
// //         (
// //             Model {
// //                 params: params.clone(),
// //                 _scenario: scenario.clone(), //TODO remove
// //                 infusions: vec![],
// //                 cov: None,
// //                 // diffeq: self.diffeq,
// //             },
// //             scenario.reorder_with_lag(vec![(0.0, 1)]),
// //         )
// //     }

// //     // This function is used to get the output from the model, defined by the output equations (outeq) supplied by the user
// //     fn get_output(&self, time: f64, x: &Self::State, system: &Self::Model, outeq: usize) -> f64 {
// //         // Get parameters from the model also used for calculating the output equations
// //         let v = system.params[2];
// //         #[allow(unused_variables)]
// //         let t = time;
// //         match outeq {
// //             1 => x[0] / v, // Concentration of the central compartment defined by the amount of drug, x[0], divided by the volume of the central compartment, v
// //             2 => x[1] / v, // Concentration of the central compartment defined by the amount of drug, x[0], divided by the volume of the central compartment, v
// //             _ => panic!("Invalid output equation"),
// //         }
// //     }

// //     // Add any possible infusions
// //     fn add_infusion(&self, system: &mut Self::Model, infusion: Infusion) {
// //         system.infusions.push(infusion);
// //     }
// //     // Add any possible covariates
// //     fn add_covs(&self, system: &mut Self::Model, cov: Option<HashMap<String, CovLine>>) {
// //         system.cov = cov;
// //     }
// //     // Add any possible doses
// //     fn add_dose(&self, state: &mut Self::State, dose: f64, compartment: usize) {
// //         state[compartment] += dose;
// //     }
// //     // Perform a "step" of the model, i.e. solve the ODEs from the current time to the next time
// //     // In the next step, we use this result as the initial state
// //     fn state_step(&self, x: &mut Self::State, system: &Self::Model, time: f64, next_time: f64) {
// //         if time >= next_time {
// //             // panic!("time error")
// //             return;
// //         }
// //         let mut stepper = Dopri5::new(system.clone(), time, next_time, 1e-3, x.clone(), RTOL, ATOL);
// //         let _res = stepper.integrate();
// //         let y = stepper.y_out();
// //         let a = y.last().unwrap();
// //         *x = a.clone();
// //     }
// // }
