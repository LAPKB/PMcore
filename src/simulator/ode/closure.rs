use std::{cell::RefCell, rc::Rc};

use diffsol::{
    jacobian::{find_non_zeros_nonlinear, JacobianColoring},
    matrix::{Matrix, MatrixSparsity},
    op::{NonLinearOp, Op, OpStatistics},
    vector::Vector,
};

use crate::prelude::data::{Covariates, Infusion};
pub struct PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    covariates: Covariates,
    infusions: Vec<Infusion>,
}

impl<M, F> PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    pub fn new(
        func: F,
        nstates: usize,
        nout: usize,
        p: Rc<M::V>,
        covariates: Covariates,
        infusions: Vec<Infusion>,
    ) -> Self {
        let nparams = p.len();
        Self {
            func,
            nstates,
            nout,
            nparams,
            p,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            covariates: covariates,
            infusions: infusions,
        }
    }
    pub fn calculate_sparsity(&mut self, y0: &M::V, t0: M::T) {
        let non_zeros = find_non_zeros_nonlinear(self, y0, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
    }
}

impl<M, F> Op for PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    type V = M::V;
    type T = M::T;
    type M = M;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
        self.sparsity.as_ref()
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F> NonLinearOp for PMClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, M::V, &Covariates),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        let mut rateiv = Self::V::zeros(self.nstates);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= Self::T::from(infusion.time)
                && t <= Self::T::from(infusion.duration + infusion.time)
            {
                rateiv[infusion.input] = Self::T::from(infusion.amount / infusion.duration);
            }
        }
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, y, rateiv, &self.covariates)
    }
    fn jac_mul_inplace(&self, _x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        let rateiv = Self::V::zeros(self.nstates);
        self.statistics.borrow_mut().increment_jac_mul();
        (self.func)(v, self.p.as_ref(), t, y, rateiv, &self.covariates);
        // (self.jacobian_action)(x, self.p.as_ref(), t, v, y)
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = self.coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
}
