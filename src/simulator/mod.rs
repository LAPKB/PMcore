pub mod likelihood;
use crate::routines::data::SubjectTrait;
use likelihood::ObsPred;
pub enum Equation {
    ODE(fn(&[f64], &[f64]) -> Vec<f64>),
    SDE(
        fn(&[f64], &[f64]) -> Vec<f64>,
        fn(&[f64], &[f64]) -> Vec<f64>,
    ),
    Analytical(fn(&[f64], &[f64]) -> Vec<f64>),
}

pub trait Simulator {
    fn new(eq: Equation) -> Self;
    fn simulate_subject(&self, subject: &impl SubjectTrait, support_point: &[f64]) -> ObsPred;
}
