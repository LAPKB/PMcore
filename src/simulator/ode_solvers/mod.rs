use ode_solvers::Dopri5;

use crate::{
    routines::data::{Covariates, Infusion},
    simulator::{DiffEq, FromVec, V},
};

const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

type State = V;
type Time = f64;

#[derive(Debug, Clone)]
struct Model {
    diffeq: DiffEq,
    support_point: Vec<f64>,
    infusions: Vec<Infusion>,
    cov: Covariates,
}
impl Model {}
impl ode_solvers::System<Time, State> for Model {
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let support_point = V::from_vec(self.support_point.clone());
        let mut rateiv = V::from_vec(vec![0.0, 0.0, 0.0]);
        //TODO: This should be pre-calculated
        for infusion in &self.infusions {
            if t >= infusion.time && t <= infusion.duration + infusion.time {
                rateiv[infusion.input] = infusion.amount / infusion.duration;
            }
        }
        (self.diffeq)(y, &support_point, t, dy, rateiv, &self.cov)
    }
}

#[inline(always)]
pub fn simulate_ode_event(
    diffeq: &DiffEq,
    x: V,
    support_point: &[f64],
    cov: &Covariates,
    infusions: &Vec<Infusion>,
    ti: f64,
    tf: f64,
) -> V {
    if ti > tf {
        panic!("time error")
    } else if ti == tf {
        return x;
    }
    let model = Model {
        diffeq: diffeq.clone(),
        support_point: support_point.to_vec(),
        infusions: infusions.clone(),
        cov: cov.clone(),
    };
    let mut stepper = Dopri5::new(model, ti, tf, 1e-3, x, RTOL, ATOL);
    let _res = stepper.integrate();
    let y = stepper.y_out();
    let a = y.last().unwrap();
    a.clone()
}
