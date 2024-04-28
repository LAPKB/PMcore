pub mod likelihood;
use crate::{
    routines::data::{Covariates, Infusion, OccasionTrait, SubjectTrait},
    simulator::{likelihood::ToObsPred, pm_ode::BuildPmOde},
};

use diffsol::{ode_solver::method::OdeSolverMethod, vector::Vector, Bdf, OdeBuilder};

use likelihood::ObsPred;
mod pm_ode;
type T = f64;
type V = faer::Col<T>;
type M = faer::Mat<T>;
pub type DiffEq = fn(&V, &V, T, &mut V, V, &Covariates);
pub type Init = fn(&V, T) -> V;
pub type Out = fn(&V, &V, T, &Covariates) -> V;

const RTOL: f64 = 1e-4;

pub enum Equation {
    ODE(DiffEq, Init, Out),
    SDE(DiffEq, DiffEq, Init, Out),
    Analytical(DiffEq),
}

impl Equation {
    pub fn new_ode(diffeq: DiffEq, init: Init, out: Out) -> Self {
        Equation::ODE(diffeq, init, out)
    }
    pub fn simulate_subject(
        &self,
        subject: &impl SubjectTrait,
        support_point: &Vec<f64>,
    ) -> Vec<ObsPred> {
        match self {
            Equation::ODE(eqn, init, out) => simulate_ode(eqn, init, out, subject, support_point),
            Equation::SDE(eqn, _, init, out) => {
                unimplemented!("Not Implemented");
            }
            Equation::Analytical(eqn) => {
                unimplemented!("Not Implemented");
            }
        }
    }
}
use crate::prelude::data::Event;
#[inline]
fn simulate_ode(
    diffeq: &DiffEq,
    init: &Init,
    out: &Out,
    subject: &impl SubjectTrait,
    support_point: &Vec<f64>,
) -> Vec<ObsPred> {
    let mut x = get_first_state(init, support_point);
    let mut infusions = vec![];
    let mut yout = vec![];
    for occasion in subject.get_occasions() {
        let covariates = occasion.get_covariates().unwrap();
        let mut index = 0;
        for event in &occasion.events {
            match event {
                Event::Bolus(bolus) => {
                    x[bolus.compartment] += bolus.amount;
                }
                Event::Infusion(infusion) => {
                    //TODO: remove not valid infusions
                    infusions.push(infusion.clone());
                }
                Event::Observation(observation) => {
                    let pred = (out)(
                        &x,
                        &V::from_vec(support_point.clone()),
                        observation.time,
                        covariates,
                    )[observation.outeq];
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
fn simulate_ode_event(
    diffeq: &DiffEq,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &Vec<Infusion>,
    ti: f64,
    tf: f64,
) -> V {
    let problem = OdeBuilder::new()
        .rtol(RTOL)
        .atol([RTOL])
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

#[inline]
fn get_first_state(init: &Init, support_point: &Vec<f64>) -> V {
    (init)(&V::from_vec(support_point.clone()), 0.0) //TODO: Time hardcoded
}
