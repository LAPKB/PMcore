#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use pmcore::{
    prelude::*,
    simulator::{analytical::one_compartment_with_absorption, Equation},
};

// // Constants for the absolute and relative tolerance for the dynamic steps used for solving the ODEs
// const ATOL: f64 = 1e-4;
// const RTOL: f64 = 1e-4;

// // Define the state vector, which must be equal to the number of compartments in the model
// // These are re-exported from the `nalgebra`-crate by `ode_solvers`, see https://github.com/srenevey/ode-solvers?tab=readme-ov-file#type-alias-definition
// // In brief, for up to 6 compartments, use VectorN<f64>, N being the number of compartments.
// // For more than 6 compartments, use `nalgebra::SVector<f64, N>`, where N is the number of compartments.
// type State = SVector<f64, 2>;
// // Time uses f64 precision
// type Time = f64;

// // This is the main structure for the model
// // It holds the parameters (defined in config.toml), the scenarios (i.e. datafile), the (possible) infusions and the covariates if any
// #[derive(Debug, Clone)]
// struct Model {
//     params: HashMap<String, f64>,
//     _scenario: Scenario,
//     infusions: Vec<Infusion>,
//     cov: Option<HashMap<String, CovLine>>,
// }
// // This is a helper function to get the parameter value by name
// impl Model {
//     pub fn get_param(&self, str: &str) -> f64 {
//         *self.params.get(str).unwrap()
//     }
// }

// impl ode_solvers::System<Time, State> for Model {
//     /// The system function, defining the ordinary differential equations (ODEs) to be solved
//     fn system(&self, t: Time, x: &State, dx: &mut State) {
//         // Get the parameters, covariates and secondary equations from the model
//         let ka = self.get_param("ka");
//         let ke = self.get_param("ke");
//         // let tlag1 = self.get_param("tlag1");
//         // let v = self.get_param("v");
//         let cov = self.cov.as_ref().unwrap();
//         let wt = match cov.get("WT") {
//             Some(x) => x.interp(t),
//             None => {
//                 dbg!(cov);
//                 panic!("WT not found in covariates")
//             }
//         };
//         // let africa = cov.get("AFRICA").unwrap().interp(t);
//         // let age = cov.get("AGE").unwrap().interp(t);
//         // let gender = cov.get("GENDER").unwrap().interp(t);
//         // let height = cov.get("HEIGHT").unwrap().interp(t);

//         let mut rateiv = [0.0]; //TODO: hardcoded
//         for infusion in &self.infusions {
//             if t >= infusion.time && t <= (infusion.dur + infusion.time) {
//                 rateiv[infusion.compartment] = infusion.amount / infusion.dur;
//             }
//         }
//         dx[0] = -ka * x[0];
//         dx[1] = ka * x[0] - ke * x[1];
//     }
// }

fn main() -> Result<()> {
    // let eq = Equation::new_ode_solvers(
    //     |x, p, _t, dx, rateiv, _cov| {
    //         // fetch_cov!(cov, t, wt);
    //         fetch_params!(p, ke, _v);
    //        dx[0] = -la * x[0];
    //        dx[1] = ka * x[0] - ke * x[1];
    //     },
    //     |p| {
    //     fetch_params!(p, _ke, _ka, _v, tlag);
    //     lag! {0=>tlag}
    // },
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ke, v);
    //         y[0] = x[1] / v;
    //     },
    //     (2, 1),
    // );
    // let eq = Equation::new_ode(
    //     |x, p, _t, dx, rateiv, _cov| {
    //         // fetch_cov!(cov, t, wt);
    //         fetch_params!(p, ke, _v);
    //        dx[0] = -la * x[0];
    //        dx[1] = ka * x[0] - ke * x[1];
    //     },
    //     |p| {
    //     fetch_params!(p, _ke, _ka, _v, tlag);
    //     lag! {0=>tlag}
    // },
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ke, v);
    //         y[0] = x[1] / v;
    //     },
    //  (2, 1),
    // );
    let eq = Equation::new_analytical(
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
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    let _result = start(eq, "examples/two_eq_lag/config.toml".to_string())?;

    Ok(())
}
