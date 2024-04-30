use diffsol::vector::Vector;
use pmcore::routines::data::{parse_pmetrics::read_pmetrics, Covariates, DataTrait};
use pmcore::simulator::analytical::{self, one_comparment};
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
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |_p: &V, _t: T| V::from_vec(vec![0.1, 0.2]),
        |x: &V, p: &V, _t: T, _cov: &Covariates| {
            fetch_params!(p, _ka, _ke, v);
            V::from_vec(vec![x[1] / v])
        },
    );

    let sim = ode.simulate_subject(first_subject, &vec![0.9, 0.1, 50.0]);
    dbg!(sim);

    let analytical = Equation::new_analytical(
        one_comparment,
        |_p: &V, _t: T| V::from_vec(vec![0.1, 0.2]),
        |x: &V, p: &V, _t: T, _cov: &Covariates| {
            fetch_params!(p, _ke, v);
            V::from_vec(vec![x[0] / v])
        },
    );

    let sim = analytical.simulate_subject(first_subject, &vec![0.9, 50.0]);
    dbg!(sim);
}
