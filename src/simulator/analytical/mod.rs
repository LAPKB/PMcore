use crate::{
    routines::data::{Covariates, OccasionTrait, SubjectTrait},
    simulator::*,
};

// let eq = |x: &V, p: &V, ti: T, tf: T, rateiv: V, _cov: &Covariates|

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
        ti,
        tf,
        rateiv,
        cov,
    )
}
