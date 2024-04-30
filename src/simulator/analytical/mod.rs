use crate::{
    routines::data::{Covariates, OccasionTrait, SubjectTrait},
    simulator::*,
};

// let eq = |x: &V, p: &V, t:T, rateiv: V, _cov: &Covariates|

#[inline]
pub fn simulate_analytical_event(
    eq: &AnalyticalEq,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &Vec<Infusion>,
    ti: f64,
    tf: f64,
) -> V {
    let mut rateiv = V::zeros(x.len());
    //TODO: This should be pre-calculated
    for infusion in infusions {
        if tf >= infusion.time && tf <= infusion.duration + infusion.time {
            rateiv[infusion.input] = infusion.amount / infusion.duration;
        }
    }
    (eq)(
        &x,
        &faer::Col::from_vec(support_point.to_vec()),
        tf - ti,
        rateiv,
        cov,
    )
}

///
/// Analytical for one compartment
/// Assumptions:
///   - p is a vector of length 1 with the value of the elimination constant
///   - rateiv is a vector of length 1 with the value of the infusion rate (only one drug)
///   - covariates are not used
///

pub fn one_compartment(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let mut xout = x.clone();
    let ke = p[0];

    xout[0] = x[0] * (-ke * t).exp() + rateiv[0] / ke * (1.0 - (-ke * t).exp());
    // dbg!(t, &rateiv, x, &xout);
    xout
}

pub fn one_compartment_with_absorption(x: &V, p: &V, t: T, rateiv: V, _cov: &Covariates) -> V {
    let mut xout = x.clone();
    let ke = p[0];
    let ka = p[1];

    xout[0] = x[0] * (-ka * t).exp();

    xout[1] = x[1] * (-ke * t).exp()
        + rateiv[0] / ke * (1.0 - (-ke * t).exp())
        + ((ka * x[0]) / (ka - ke)) * ((-ke * t).exp() - (-ka * t).exp());

    xout
}
