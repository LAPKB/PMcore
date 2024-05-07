use crate::routines::data::{Covariates, Infusion};
use diffsol::{
    matrix::Matrix,
    ode_solver::problem::OdeSolverProblem,
    op::{unit::UnitCallable, Op},
    vector::Vector,
    OdeBuilder, OdeEquations, Result, Zero,
};
use std::rc::Rc;

use super::closure::PMClosure;

pub struct OdePmSolverEquationsMassI<M, F, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: Rc<PMClosure<M, F>>,
    mass: Rc<UnitCallable<M>>,
    init: I,
    p: Rc<M::V>,
}

impl<M, F, I> OdePmSolverEquationsMassI<M, F, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_pm_ode(
        rhs: F,
        init: I,
        p: M::V,
        t0: M::T,
        calculate_sparsity: bool,
        covariates: Covariates,
        infusions: Vec<Infusion>,
    ) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);
        let mut rhs = PMClosure::new(rhs, nstates, nstates, p.clone(), covariates, infusions);
        if calculate_sparsity {
            rhs.calculate_sparsity(&y0, t0);
        }
        let mass = UnitCallable::<M>::new(nstates);
        let rhs = Rc::new(rhs);
        let mass = Rc::new(mass);
        Self {
            rhs,
            mass,
            init,
            p: p.clone(),
        }
    }
}

impl<M, F, I> OdeEquations for OdePmSolverEquationsMassI<M, F, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    type T = M::T;
    type V = M::V;
    type M = M;
    type Rhs = PMClosure<M, F>;
    type Mass = UnitCallable<M>;
    fn mass(&self) -> &Rc<Self::Mass> {
        &self.mass
    }

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }

    fn is_mass_constant(&self) -> bool {
        true
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }
}

pub trait BuildPmOde {
    fn build_pm_ode<M, F, I>(
        self,
        rhs: F,
        init: I,
        cov: Covariates,
        infusions: Vec<Infusion>,
    ) -> Result<OdeSolverProblem<OdePmSolverEquationsMassI<M, F, I>>>
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
        I: Fn(&M::V, M::T) -> M::V;
}

impl BuildPmOde for OdeBuilder {
    fn build_pm_ode<M, F, I>(
        self,
        rhs: F,
        init: I,
        cov: Covariates,
        infusions: Vec<Infusion>,
    ) -> Result<OdeSolverProblem<OdePmSolverEquationsMassI<M, F, I>>>
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Self::build_p(self.p);
        let eqn = OdePmSolverEquationsMassI::new_pm_ode(
            rhs,
            init,
            p,
            M::T::from(self.t0),
            self.use_coloring,
            cov,
            infusions,
        );

        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        Ok(OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
        ))
    }
}
