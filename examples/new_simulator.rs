use diffsol::vector::Vector;
use pmcore::routines::data::{parse_pmetrics::read_pmetrics, Covariates, DataTrait};
use pmcore::simulator::analytical::{one_compartment, one_compartment_with_absorption};
use pmcore::{prelude::*, simulator::Equation};
use std::path::Path;
type T = f64;
type V = faer::Col<T>;

fn main() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();

    let ode = Equation::new_ode(
        |x: &V, p: &V, _t: T, dx: &mut V, rateiv: V, _cov: &Covariates| {
            //fetch_cov!(cov, t, creat);
            fetch_params!(p, ke, ka, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |_p: &V, _t: T, cov: &Covariates| V::from_vec(vec![0.1, 0.2]),
        |x: &V, p: &V, _t: T, _cov: &Covariates| {
            fetch_params!(p, _ke, _ka, v);
            V::from_vec(vec![x[1] / v])
        },
    );

    let sim_ode = ode.simulate_subject(first_subject, &vec![0.1, 0.9, 50.0]);

    let analytical = Equation::new_analytical(
        one_compartment_with_absorption,
        |_p: &V, _t: T, _cov: &Covariates| V::from_vec(vec![0.1, 0.2]),
        |x: &V, p: &V, _t: T, _cov: &Covariates| {
            fetch_params!(p, _ke, _ka, v);
            V::from_vec(vec![x[1] / v])
        },
    );

    let sim_analytical = analytical.simulate_subject(first_subject, &vec![0.1, 0.9, 50.0]);
    sim_ode.iter().zip(&sim_analytical).for_each(|(ode, anal)| {
        println!("ode: {:?}", ode);
        println!("anal: {:?}", anal);
    })
}
