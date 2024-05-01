use faer::col;
use pmcore::routines::data::{parse_pmetrics::read_pmetrics, DataTrait};
use pmcore::simulator::analytical::one_compartment_with_absorption;
use pmcore::{prelude::*, simulator::Equation};
use std::path::Path;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench()]
pub fn ode() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();

    let ode = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            //fetch_cov!(cov, t, creat);
            fetch_params!(p, ke, ka, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |_p, _t, cov| col![0.1, 0.2],
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v);
            col![x[1] / v]
        },
    );

    for _ in 0..100 {
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
        |_p, _t, _cov| col![0.1, 0.2],
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v);
            col![x[1] / v]
        },
    );
    for _ in 0..100 {
        let _ = analytical.simulate_subject(first_subject, &vec![0.1, 0.9, 50.0]);
    }
}
