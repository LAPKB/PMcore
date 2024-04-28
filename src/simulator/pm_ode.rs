use crate::routines::data::{
    parse_pmetrics::read_pmetrics, CovariateTrait, Covariates, CovariatesTrait, DataTrait,
    Infusion, OccasionTrait,
};
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
use std::{cell::RefCell, path::Path, rc::Rc};

pub struct OdePmSolverEquationsMassI<M, F, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: F,
    init: I,
    p: Rc<M::V>,
    nstates: usize,
    coloring: Option<JacobianColoring>,
    statistics: RefCell<OdeEquationsStatistics>,
    covariates: Covariates,
    infusions: Vec<Infusion>,
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
        use_coloring: bool,
        covariates: Covariates,
        infusions: Vec<Infusion>,
    ) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);

        let statistics = RefCell::default();
        let mut ret = Self {
            rhs,
            // rhs_jac,
            init,
            p: p.clone(),
            nstates,
            coloring: None,
            statistics,
            covariates,
            infusions,
        };
        let coloring = if use_coloring {
            let rhs_inplace = |x: &M::V, _p: &M::V, t: M::T, y_rhs: &mut M::V| {
                ret.rhs_inplace(t, x, y_rhs);
            };
            let rhs_jac_inplace = |x: &M::V, _p: &M::V, t: M::T, v: &M::V, y: &mut M::V| {
                ret.rhs_jac_inplace(t, x, v, y);
            };
            let op =
                Closure::<M, _, _>::new(rhs_inplace, rhs_jac_inplace, nstates, nstates, p.clone());
            Some(JacobianColoring::new(&op, &y0, t0))
        } else {
            None
        };
        ret.coloring = coloring;
        ret
    }
}

// impl Op
impl<M, F, I> Op for OdePmSolverEquationsMassI<M, F, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.p.len()
    }
    fn nstates(&self) -> usize {
        self.nstates
    }
}

impl<M, F, I> OdeEquations for OdePmSolverEquationsMassI<M, F, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
    I: Fn(&M::V, M::T) -> M::V,
{
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        let p = self.p.as_ref();
        let mut rateiv = Self::V::zeros(self.nstates);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= Self::T::from(infusion.time)
                && t <= Self::T::from(infusion.duration + infusion.time)
            {
                rateiv[infusion.compartment] = Self::T::from(infusion.amount / infusion.duration);
            }
        }
        (self.rhs)(y, p, t, rhs_y, rateiv, &self.covariates);
        self.statistics.borrow_mut().number_of_rhs_evals += 1;
    }

    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        let mut rateiv = Self::V::zeros(self.nstates);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= Self::T::from(infusion.time)
                && t <= Self::T::from(infusion.duration + infusion.time)
            {
                rateiv[infusion.compartment] = Self::T::from(infusion.amount / infusion.duration);
            }
        }
        (self.rhs)(v, p, t, y, rateiv, &self.covariates);
        // (self.rhs_jac)(x, p, t, v, y);
        self.statistics.borrow_mut().number_of_jac_mul_evals += 1;
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }

    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        <Self::V as Vector>::Index::zeros(0)
    }

    fn jacobian_matrix(&self, x: &Self::V, t: Self::T) -> Self::M {
        self.statistics.borrow_mut().number_of_jacobian_matrix_evals += 1;
        let rhs_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y_rhs: &mut Self::V| {
            self.rhs_inplace(t, x, y_rhs);
        };
        let rhs_jac_inplace =
            |x: &Self::V, _p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V| {
                self.rhs_jac_inplace(t, x, v, y);
            };
        let op = Closure::<M, _, _>::new(
            rhs_inplace,
            rhs_jac_inplace,
            self.nstates,
            self.nstates,
            self.p.clone(),
        );
        let triplets = if let Some(coloring) = &self.coloring {
            coloring.find_non_zero_entries(&op, x, t)
        } else {
            find_non_zero_entries(&op, x, t)
        };
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }
    fn get_statistics(&self) -> OdeEquationsStatistics {
        self.statistics.borrow().clone()
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
        let atol = Self::build_atol(self.atol, eqn.nstates())?;
        Ok(OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
        ))
    }
}

macro_rules! fetch_params {
    ($p:expr, $($name:ident),*) => {
        let p = $p;
        let mut idx = 0;
        $(
            let $name = p.get(idx);
            idx += 1;
        )*
    };
}

macro_rules! fetch_cov {
    ($cov:expr, $t:expr, $($name:ident),*) => {
        $(
            let $name = $cov.get_covariate(stringify!($name)).unwrap().interpolate($t).unwrap();
        )*
    };
}
