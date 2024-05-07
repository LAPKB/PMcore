use crate::prelude::data::{Covariates, Infusion};
use anyhow::{Ok, Result};
use diffsol::{
    matrix::Matrix,
    ode_solver::{equations::OdeSolverEquations, problem::OdeSolverProblem},
    op::{unit::UnitCallable, Op},
    vector::Vector,
    OdeEquations,
};

use std::rc::Rc;

use super::closure::PMClosure;

pub fn build_pm_ode<M, F, I, Ite, T>(
    rhs: F,
    init: I,
    p: M::V,
    t0: f64,
    h0: f64,
    rtol: f64,
    atol: Ite,
    cov: Covariates,
    infusions: Vec<Infusion>,
) -> Result<OdeSolverProblem<OdeSolverEquations<M, PMClosure<M, F>, I>>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
    Ite: IntoIterator<Item = T>,
    f64: From<T>,
{
    let p = Rc::new(p);
    let t0 = M::T::from(t0);
    let y0 = (init)(&p, t0);
    let nstates = y0.len();
    let rhs = PMClosure::new(rhs, nstates, nstates, p.clone(), cov, infusions);
    let mass = Rc::new(UnitCallable::new(nstates));
    let rhs = Rc::new(rhs);
    let eqn = OdeSolverEquations::new(rhs, mass, None, init, p, false);
    let atol = atol.into_iter().map(|x| f64::from(x)).collect();
    let atol = build_atol(atol, eqn.rhs().nstates())?;
    Ok(OdeSolverProblem::new(
        eqn,
        M::T::from(rtol),
        atol,
        t0,
        M::T::from(h0),
    ))
}
fn build_atol<V: Vector>(atol: Vec<f64>, nstates: usize) -> Result<V> {
    if atol.len() == 1 {
        Ok(V::from_element(nstates, V::T::from(atol[0])))
    } else if atol.len() != nstates {
        Err(anyhow::anyhow!(
            "atol must have length 1 or equal to the number of states"
        ))
    } else {
        let mut v = V::zeros(nstates);
        for (i, &a) in atol.iter().enumerate() {
            v[i] = V::T::from(a);
        }
        Ok(v)
    }
}
