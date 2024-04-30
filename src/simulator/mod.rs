pub mod analytical;
pub mod likelihood;
pub mod ode;
use crate::{
    prelude::data::Event,
    routines::data::{Covariates, Infusion, OccasionTrait, SubjectTrait},
    simulator::likelihood::{ObsPred, ToObsPred},
};

pub type T = f64;
pub type V = faer::Col<T>;
pub type M = faer::Mat<T>;

pub type DiffEq = fn(&V, &V, T, &mut V, V, &Covariates);
pub type Init = fn(&V, T) -> V;
pub type Out = fn(&V, &V, T, &Covariates) -> V;
pub type AnalyticalEq = fn(&V, &V, T, &mut V, V, &Covariates) -> V;

pub enum Equation {
    ODE(DiffEq, Init, Out),
    SDE(DiffEq, DiffEq, Init, Out),
    Analytical(AnalyticalEq, Init, Out),
}

impl Equation {
    pub fn new_ode(diffeq: DiffEq, init: Init, out: Out) -> Self {
        Equation::ODE(diffeq, init, out)
    }
    pub fn new_analytical(eq: AnalyticalEq, init: Init, out: Out) -> Self {
        Equation::Analytical(eq, init, out)
    }

    pub fn simulate_subject(
        &self,
        subject: &impl SubjectTrait,
        support_point: &Vec<f64>,
    ) -> Vec<ObsPred> {
        let init = self.get_init();
        let out = self.get_out();
        let mut yout = vec![];
        for occasion in subject.get_occasions() {
            // What should we use as the initial state for the next occasion?
            let mut x = get_first_state(init, support_point);
            let covariates = occasion.get_covariates().unwrap();
            let mut infusions: Vec<Infusion> = vec![];
            let mut index = 0;
            for event in &occasion.get_events(None, None, true) {
                match event {
                    Event::Bolus(bolus) => {
                        x[bolus.input] += bolus.amount;
                    }
                    Event::Infusion(infusion) => {
                        infusions.push(infusion.clone());
                    }
                    Event::Observation(observation) => {
                        let pred = (out)(
                            &x,
                            &V::from_vec(support_point.clone()),
                            observation.time,
                            covariates,
                        )[observation.outeq - 1];

                        yout.push(observation.to_obs_pred(pred));
                    }
                }
                if let Some(next_event) = occasion.events.get(index + 1) {
                    x = self.simulate_event(
                        x,
                        support_point,
                        covariates,
                        &infusions,
                        event.get_time(),
                        next_event.get_time(),
                    );
                }
                index += 1;
            }
        }
        yout
        // match self {
        //     Equation::ODE(eqn, init, out) => simulate_ode(eqn, init, out, subject, support_point),
        //     Equation::SDE(_eqn, _, _init, _out) => {
        //         unimplemented!("Not Implemented");
        //     }
        //     Equation::Analytical(_eqn, _init, _out) => {
        //         unimplemented!("Not Implemented");
        //     }
        // }
    }
    fn simulate_event(
        &self,
        x: V,
        support_point: &Vec<f64>,
        covariates: &Covariates,
        infusions: &Vec<Infusion>,
        start_time: T,
        end_time: T,
    ) -> V {
        match self {
            Equation::ODE(eqn, init, out) => ode::simulate_ode_event(
                eqn,
                x,
                support_point,
                covariates,
                infusions,
                start_time,
                end_time,
            ),
            Equation::SDE(_, _, _, _) => {
                unimplemented!("Not Implemented");
            }
            Equation::Analytical(_, _, _) => {
                unimplemented!("Not Implemented");
            }
        }
    }
    fn get_init(&self) -> &Init {
        match self {
            Equation::ODE(_, init, _) => init,
            Equation::SDE(_, _, init, _) => init,
            Equation::Analytical(_, init, _) => init,
        }
    }
    fn get_out(&self) -> &Out {
        match self {
            Equation::ODE(_, _, out) => out,
            Equation::SDE(_, _, _, out) => out,
            Equation::Analytical(_, _, out) => out,
        }
    }
}

#[inline]
pub fn get_first_state(init: &Init, support_point: &Vec<f64>) -> V {
    (init)(&V::from_vec(support_point.clone()), 0.0) //TODO: Time hardcoded
}

trait FromVec {
    fn from_vec(v: Vec<f64>) -> Self;
}

impl FromVec for V {
    fn from_vec(vec: Vec<T>) -> Self {
        V::from_fn(vec.len(), |i| vec[i])
    }
}
