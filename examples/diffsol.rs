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
};
use std::{cell::RefCell, path::Path, rc::Rc};

pub struct OdePmSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, &Covariates),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: F,
    rhs_jac: G,
    init: I,
    p: Rc<M::V>,
    nstates: usize,
    coloring: Option<JacobianColoring>,
    statistics: RefCell<OdeEquationsStatistics>,
    covariates: Covariates,
}

impl<M, F, G, I> OdePmSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, &Covariates),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_pm_ode(
        rhs: F,
        rhs_jac: G,
        init: I,
        p: M::V,
        t0: M::T,
        use_coloring: bool,
        covariates: Covariates,
    ) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);

        let statistics = RefCell::default();
        let mut ret = Self {
            rhs,
            rhs_jac,
            init,
            p: p.clone(),
            nstates,
            coloring: None,
            statistics,
            covariates,
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
impl<M, F, G, I> Op for OdePmSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, &Covariates),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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

impl<M, F, G, I> OdeEquations for OdePmSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V, &Covariates),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs)(y, p, t, rhs_y, &self.covariates);
        self.statistics.borrow_mut().number_of_rhs_evals += 1;
    }

    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs_jac)(x, p, t, v, y);
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

trait BuildPmOde {
    fn build_pm_ode<M, F, G, I>(
        self,
        rhs: F,
        rhs_jac: G,
        init: I,
        cov: Covariates,
    ) -> Result<OdeSolverProblem<OdePmSolverEquationsMassI<M, F, G, I>>>
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V, &Covariates),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V;
}

impl BuildPmOde for OdeBuilder {
    fn build_pm_ode<M, F, G, I>(
        self,
        rhs: F,
        rhs_jac: G,
        init: I,
        cov: Covariates,
    ) -> Result<OdeSolverProblem<OdePmSolverEquationsMassI<M, F, G, I>>>
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V, &Covariates),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Self::build_p(self.p);
        let eqn = OdePmSolverEquationsMassI::new_pm_ode(
            rhs,
            rhs_jac,
            init,
            p,
            M::T::from(self.t0),
            self.use_coloring,
            cov,
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
    (($($name:ident),*)) => {
        let ($(ref $name),*) = ($(p[$crate::destructure_array::index!()]),*);
    };
}

macro_rules! index {
    () => { 0 };
    ( $($other:tt)* ) => { 1 + $crate::destructure_array::index!($($other)*) };
}
fn main() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke_blocks.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = subjects.first().unwrap();
    let occasion = first_subject.occasions.first().unwrap();
    let covariates = occasion.covariates.clone();
    println!("{}", data);

    type T = f64;
    type V = faer::Col<T>;
    type M = faer::Mat<T>;
    // type V = nalgebra::DVector<T>;
    // type M = nalgebra::DMatrix<T>;
    let problem = OdeBuilder::new()
        .p([0.04, 1.0e4, 3.0e7])
        .rtol(1e-4)
        .atol([1.0e-8, 1.0e-6])
        .build_pm_ode::<M, _, _, _>(
            |x: &V, p: &V, t: T, dx: &mut V, cov: &Covariates| {
                let creat = cov.get_covariate("creat").unwrap().interpolate(t).unwrap();
                fetch_params!((ka, ke));
                dx[0] = -ka * x[0];
                dx[1] = ka * x[0] - ke * x[1];
            },
            |_x: &V, p: &V, _t: T, v: &V, dx: &mut V| {
                dx[0] = -p[0] * v[0];
                dx[1] = p[0] * v[0] - p[1] * v[1];
            },
            |_p: &V, _t: T| V::from_vec(vec![1.0, 0.0]),
            covariates,
        )
        .unwrap();

    let mut solver = Bdf::default();

    let t = 0.4;
    let y = solver.solve(&problem, t).unwrap();
    dbg!(&y);
}

// we have a function is going to return a minimal closure
// this function
