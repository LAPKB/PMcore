use diffsol::{
    jacobian::{find_non_zero_entries, JacobianColoring},
    matrix::Matrix,
    ode_solver::equations::OdeEquationsStatistics,
    ode_solver::method::OdeSolverMethod,
    ode_solver::problem::OdeSolverProblem,
    op::{closure::Closure, Op},
    vector::{Vector, VectorIndex},
    Bdf, OdeBuilder, OdeEquations, Result, Zero,
};
use pmcore::routines::data::{
    parse_pmetrics::read_pmetrics, CovariateTrait, Covariates, CovariatesTrait, DataTrait,
    Infusion, OccasionTrait,
};
use pmcore::{prelude::*, simulator::Equation};
use std::{cell::RefCell, path::Path, rc::Rc};
fn main() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke_blocks.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();
    let occasion = first_subject.occasions.first().unwrap();
    let covariates = occasion.covariates.clone();
    let infusions = occasion.get_infusions();
    println!("{}", data);

    type T = f64;
    type V = faer::Col<T>;
    type M = faer::Mat<T>;
    let diffeq = |x: &V, p: &V, t: T, dx: &mut V, rateiv: V, cov: &Covariates| {
        fetch_cov!(cov, t, creat);
        fetch_params!(p, ka, ke, _v);
        dx[0] = -ka * x[0];
        dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
    };
    let init = |_p: &V, _t: T| V::from_vec(vec![0.1, 0.2]);
    let out = |x: &V, p: &V, _t: T, _cov: &Covariates| {
        fetch_params!(p, _ka, _ke, v);
        V::from_vec(vec![x[1] / v])
    };
    let ode = Equation::new_ode(diffeq, init, out);

    let sim = ode.simulate_subject(first_subject, &vec![0.9, 0.1, 50.0]);
    dbg!(sim);
}
