use std::collections::HashMap;

use ode_solvers::Vector1;

use crate::routines::datafile::{CovLine, Infusion};

pub fn one_compartment(
    x: &Vector1<f64>,
    params: &HashMap<String, f64>,
    infusions: &Vec<Infusion>,
    cov: &Option<HashMap<String, CovLine>>,
    t: f64,
    p_t: f64,
) -> Vector1<f64> {
    // let ka = params.get("ka").unwrap();
    let ke = params.get("ke").unwrap();
    let mut r = 0.0;
    for infusion in infusions {
        if t >= infusion.time && t <= (infusion.dur + infusion.time) {
            r += infusion.amount / infusion.dur;
        }
    }
    let t = t - p_t;
    Vector1::new(x[0] * (-ke * t).exp() + (r / ke) * (1.0 - (-ke * t).exp()))
}
