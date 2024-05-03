use crate::prelude::*;
use argmin::core::CostFunction;
use datafile::Scenario;
use ndarray::{Array1, Array2};
use sigma::ErrorPoly;

struct GamLam<'a> {
    obs_pred: &'a ObsPred,
    scenarios: &'a Vec<Scenario>,
    ep: ErrorPoly<'a>,
}

impl<'a> CostFunction for GamLam<'a> {
    type Param = f64;
    type Output = f64;
    fn cost(&self, _param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let prob = self.obs_pred.likelihood(&self.ep);
        Ok(prob.sum())
    }
}
