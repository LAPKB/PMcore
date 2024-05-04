use pmcore::prelude::*;
use pmcore::routines::data::{parse_pmetrics::read_pmetrics, DataTrait};
use pmcore::routines::datafile::parse;
use pmcore::simulator::analytical::one_compartment_with_absorption;
use pmcore::simulator::Equation;
use pmcore::simulator::V;
use std::path::Path;

const DATA_PATH: &str = "examples/data/two_eq_lag.csv";

fn main() {
    let data = read_pmetrics(Path::new(DATA_PATH)).unwrap();
    let subjects = data.get_subjects();
    let first_subject = subjects.first().unwrap();

    let data = parse(&DATA_PATH.to_string()).unwrap();
    let first_scenario = data.first().unwrap();

    let spp = vec![
        0.022712449789047243, //ke
        0.48245882034301757,  //ka
        71.28352475166321,    //v
        // 0.5903420448303222,   //tlag
        0.0,
    ];

    let diffsol = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, ka, _v, _tlag);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |p| {
            fetch_params!(p, _ke, _ka, _v, tlag);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v, _tlag);
            V::from_vec(vec![x[1] / v])
        },
    );

    let analytical = Equation::new_analytical(
        one_compartment_with_absorption,
        |_p, _cov| {},
        |p| {
            fetch_params!(p, _ke, _ka, _v, tlag);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v, _tlag);
            V::from_vec(vec![x[1] / v])
        },
    );

    let ode_solvers = Equation::new_ode_solvers(
        |x, p, _t, dx, rateiv, _cov| {
            //fetch_cov!(cov, t, creat);
            fetch_params!(p, ke, ka, _v, _tlag);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |p| {
            fetch_params!(p, _ke, _ka, _v, tlag);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v, _tlag);
            V::from_vec(vec![x[1] / v])
        },
    );

    let sim_diffsol_new = diffsol.simulate_subject(&first_subject, &spp);
    let sim_diffsol_old = diffsol.simulate_scenario(first_scenario, &spp);
    let sim_ode_solvers_new = ode_solvers.simulate_subject(&first_subject, &spp);
    let sim_ode_solvers_old = ode_solvers.simulate_scenario(first_scenario, &spp);
    let sim_analytical_new = analytical.simulate_subject(&first_subject, &spp);
    let sim_analytical_old = analytical.simulate_scenario(first_scenario, &spp);

    sim_diffsol_new
        .get_predictions()
        .iter()
        .zip(sim_diffsol_old.get_predictions())
        .zip(sim_analytical_new.get_predictions())
        .zip(sim_analytical_old.get_predictions())
        .zip(sim_ode_solvers_new.get_predictions())
        .zip(sim_ode_solvers_old.get_predictions())
        .for_each(|(((((dsn, dso), an), ao), osn), oso)| {
            println!("Old Simulator: ");
            println!("  diffsol : {}", dso);
            println!("  oso_sol : {}", oso);
            println!("  analytic: {}", ao);
            println!("New Simulator: ");
            println!("  diffsol : {}", dsn);
            println!("  osn_sol : {}", osn);
            println!("  analytic: {}", an);
            println!("=======================");
        })
}
