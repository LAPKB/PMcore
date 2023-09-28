use crate::prelude::*;
use argmin::core::CostFunction;
use datafile::Scenario;
use ndarray::{Array1, Array2};
use sigma::ErrorPoly;

struct GamLam<'a> {
    pred: &'a Array2<Array1<f64>>,
    scenarios: &'a Vec<Scenario>,
    ep: ErrorPoly<'a>,
}

impl<'a> CostFunction for GamLam<'a> {
    type Param = f64;
    type Output = f64;
    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let prob = prob::calculate_psi(self.pred, self.scenarios, &self.ep);
        Ok(prob.sum())
    }
}
