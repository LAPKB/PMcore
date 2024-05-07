pub mod closure;
pub mod diffsol_traits;
use crate::{
    prelude::data::{Covariates, Infusion},
    simulator::{ode::diffsol_traits::BuildPmOde, DiffEq, M, T, V},
};

use diffsol::{ode_solver::method::OdeSolverMethod, Bdf, OdeBuilder};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

#[inline(always)]
pub fn simulate_ode_event(
    diffeq: &DiffEq,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &Vec<Infusion>,
    ti: f64,
    tf: f64,
) -> V {
    let problem = OdeBuilder::new()
        .t0(ti)
        .rtol(RTOL)
        .atol([ATOL])
        .p(support_point.to_owned())
        .build_pm_ode::<M, _, _>(
            diffeq.clone(),
            move |_p: &V, _t: T| x.clone(),
            cov.clone(),
            infusions.clone(),
        )
        .unwrap();

    let mut solver = Bdf::default();
    solver.solve(&problem, tf).unwrap()
}
