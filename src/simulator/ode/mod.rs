pub mod diffsol_traits;
use crate::{
    prelude::data::Event,
    routines::data::{Covariates, Infusions, OccasionTrait, SubjectTrait},
    simulator::{
        get_first_state,
        likelihood::{ObsPred, ToObsPred},
        ode::diffsol_traits::BuildPmOde,
        DiffEq, Init, Out, M, T, V,
    },
};

use diffsol::{ode_solver::method::OdeSolverMethod, vector::Vector, Bdf, OdeBuilder};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;
#[inline]
pub fn simulate_ode(
    diffeq: &DiffEq,
    init: &Init,
    out: &Out,
    subject: &impl SubjectTrait,
    support_point: &Vec<f64>,
) -> Vec<ObsPred> {
    let mut yout = vec![];
    for occasion in subject.get_occasions() {
        // What should we use as the initial state for the next occasion?
        let mut x = get_first_state(init, support_point);
        let mut infusions: Infusions = Infusions::new();
        let covariates = occasion.get_covariates().unwrap();
        let mut index = 0;
        for event in &occasion.get_events(None, None, true) {
            match event {
                Event::Bolus(bolus) => {
                    x[bolus.compartment] += bolus.amount;
                }
                Event::Infusion(infusion) => {
                    infusions.add_infusion(infusion.clone());
                }
                Event::Observation(observation) => {
                    let pred = (out)(
                        &x,
                        &V::from_vec(support_point.clone()),
                        observation.time,
                        covariates,
                    )[observation.outeq - 1];

                    yout.push(observation.to_obs_pred(pred));
                }
            }
            if let Some(next_event) = occasion.events.get(index + 1) {
                x = simulate_ode_event(
                    diffeq,
                    x,
                    support_point,
                    covariates,
                    &infusions,
                    event.get_time(),
                    next_event.get_time(),
                );
            }
            index += 1;
        }
    }
    yout
}

#[inline]
pub fn simulate_ode_event(
    diffeq: &DiffEq,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &Infusions,
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
