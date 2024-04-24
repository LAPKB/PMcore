pub mod likelihood;
use crate::routines::data::SubjectTrait;
use likelihood::ObsPred;
pub enum Equation {
    ODE(DiffEq, Init, Out),
    SDE(DiffEq, DiffEq, Init, Out),
    Analytical(DiffEq),
}
//|x: &V, p: &V, _t: T, y: &mut V|
pub type DiffEq = fn(&[f64], &[f64], f64, &mut [f64]);
//|p: &V, t: T| -> V
pub type Init = fn(&[f64], f64) -> [f64];
//|p: &V, t: T| -> V
pub type Out = fn(&[f64], f64) -> [f64];
//|p: &V, t: T| -> V

impl Equation {
    pub fn new_ode(eqn: DiffEq, init: Init, out: Out) -> Self {
        Equation::ODE(eqn, init, out)
    }
    fn simulate_subject(&self, subject: &impl SubjectTrait, support_point: &[f64]) -> ObsPred {
        match self {
            Equation::ODE(eqn, init, out) => {
                unimplemented!("Not Implemented");
            }
            Equation::SDE(eqn, _, init, out) => {
                unimplemented!("Not Implemented");
            }
            Equation::Analytical(eqn) => {
                unimplemented!("Not Implemented");
            }
        }
    }
}
