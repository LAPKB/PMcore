pub mod analytical;
pub mod likelihood;
// pub mod ode;
pub mod ode_solvers;
pub mod cache;
use std::collections::HashMap;

use self::likelihood::SubjectPredictions;
use crate::{
    prelude::data::Event,
    routines::{
        data::{Covariates, Infusion, OccasionTrait, Subject, SubjectTrait},
        datafile::Scenario,
    },
    simulator::likelihood::{Prediction, ToPrediction},
};
// use diffsol::vector::Vector;

use cache::*;

pub type T = f64;
// pub type V = faer::Col<T>;
// pub type M = faer::Mat<T>;
// pub type V = nalgebra::DVector<T>;
pub type V = nalgebra::SVector<T, 2>;
pub type M = nalgebra::DMatrix<T>;

pub type DiffEq = fn(&V, &V, T, &mut V, V, &Covariates);
pub type Init = fn(&V, T, &Covariates) -> V;
pub type Out = fn(&V, &V, T, &Covariates) -> V;
pub type AnalyticalEq = fn(&V, &V, T, V, &Covariates) -> V;
pub type SecEq = fn(&mut V, &Covariates);
pub type Lag = fn(&V) -> HashMap<usize, T>;
pub type Fa = fn(&V) -> HashMap<usize, T>;

#[derive(Debug, Clone)]
pub enum Equation {
    OdeSolvers(DiffEq, Lag, Fa, Init, Out),
    ODE(DiffEq, Lag, Fa, Init, Out),
    SDE(DiffEq, DiffEq, Lag, Fa, Init, Out),
    Analytical(AnalyticalEq, SecEq, Lag, Fa, Init, Out),
}

impl Equation {
    pub fn new_ode(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Equation::ODE(diffeq, lag, fa, init, out)
    }
    pub fn new_analytical(
        eq: AnalyticalEq,
        seq_eq: SecEq,
        lag: Lag,
        fa: Fa,
        init: Init,
        out: Out,
    ) -> Self {
        Equation::Analytical(eq, seq_eq, lag, fa, init, out)
    }

    pub fn new_ode_solvers(diffeq: DiffEq, lag: Lag, fa: Fa, init: Init, out: Out) -> Self {
        Equation::OdeSolvers(diffeq, lag, fa, init, out)
    }

    pub fn simulate_scenario(
        &self,
        scenario: &Scenario,
        support_point: &Vec<f64>,
    ) -> SubjectPredictions {
        let init = self.get_init();
        let out = self.get_out();
        let covariates = Covariates::new();
        let lag_hashmap = self.get_lag(support_point);
        let lag = lag_hashmap
            .into_iter()
            .map(|(id, score)| (score, id))
            .collect();
        let fa_hashmap = self.get_fa(support_point);
        let scenario = &scenario.reorder_with_lag(lag);
        let mut yout = vec![];
        let mut x = (init)(&V::from_vec(support_point.clone()), 0.0, &covariates);
        let mut index = 0;
        let mut infusions = vec![];

        for block in &scenario.blocks {
            //todo: add covs
            for event in &block.events {
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        infusions.push(Infusion {
                            time: event.time,
                            duration: event.dur.unwrap(),
                            amount: event.dose.unwrap(),
                            input: event.input.unwrap() - 1,
                        })
                    } else {
                        //dose
                        let comparment = event.input.unwrap() - 1;
                        x[comparment] +=
                            event.dose.unwrap() * fa_hashmap.get(&comparment).unwrap_or(&1.0);
                    }
                } else if event.evid == 0 {
                    //obs
                    let pred = (out)(
                        &x,
                        &V::from_vec(support_point.clone()),
                        event.time,
                        &covariates,
                    )[event.outeq.unwrap() - 1];
                    // .get(event.outeq.unwrap() - 1)
                    // .unwrap();
                    yout.push(Prediction {
                        time: event.time,
                        observation: event.out.unwrap(),
                        prediction: pred,
                        outeq: event.outeq.unwrap() - 1,
                        errorpoly: None,
                    });
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    x = self.simulate_event(
                        x,
                        support_point,
                        &covariates,
                        &infusions,
                        event.time,
                        *next_time,
                    );
                }
                index += 1;
            }
        }
        yout.into()
    }

    pub fn simulate_subject(
        &self,
        subject: &Subject,
        support_point: &Vec<f64>,
    ) -> SubjectPredictions {
        let init = self.get_init();
        let out = self.get_out();
        let lag = self.get_lag(support_point);
        let fa = self.get_fa(support_point);
        let mut yout = vec![];
        for occasion in subject.get_occasions() {
            // Check for a cache entry
            let pred = get_entry(&subject.id, &support_point);
            if let Some(pred) = pred {
                return pred;
            }
            // What should we use as the initial state for the next occasion?
            let covariates = occasion.get_covariates().unwrap();
            let mut x = get_first_state(init, support_point, &covariates);
            let mut infusions: Vec<Infusion> = vec![];
            let mut index = 0;
            let events = occasion.get_events(Some(&lag), Some(&fa), true);
            for event in &events {
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
                        )[observation.outeq];

                        yout.push(observation.to_obs_pred(pred));
                    }
                }

                if let Some(next_event) = events.get(index + 1) {
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
        // Insert the cache entry
        let pred: SubjectPredictions = yout.into();
        insert_entry(&subject.id, &support_point, pred.clone());
        pred

    }
    #[inline(always)]

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
            Equation::ODE(eqn, _, _, _, _) => {
                unimplemented!("Not implemented");
                // ode::simulate_ode_event(
                //     eqn,
                //     x,
                //     support_point,
                //     covariates,
                //     infusions,
                //     start_time,
                //     end_time,
                // )
            }
            Equation::OdeSolvers(eqn, _, _, _, _) => ode_solvers::simulate_ode_event(
                eqn,
                x,
                support_point,
                covariates,
                infusions,
                start_time,
                end_time,
            ),
            Equation::SDE(_, _, _, _, _, _) => {
                unimplemented!("Not Implemented");
            }
            Equation::Analytical(eq, seq_eq, _, _, _, _) => analytical::simulate_analytical_event(
                &eq,
                &seq_eq,
                x,
                support_point,
                covariates,
                infusions,
                start_time,
                end_time,
            ),
        }
    }
    #[inline(always)]
    fn get_init(&self) -> &Init {
        match self {
            Equation::ODE(_, _, _, init, _) => init,
            Equation::OdeSolvers(_, _, _, init, _) => init,
            Equation::SDE(_, _, _, _, init, _) => init,
            Equation::Analytical(_, _, _, _, init, _) => init,
        }
    }
    #[inline(always)]
    fn get_out(&self) -> &Out {
        match self {
            Equation::ODE(_, _, _, _, out) => out,
            Equation::OdeSolvers(_, _, _, _, out) => out,
            Equation::SDE(_, _, _, _, _, out) => out,
            Equation::Analytical(_, _, _, _, _, out) => out,
        }
    }
    #[inline(always)]
    fn get_lag(&self, spp: &Vec<f64>) -> HashMap<usize, f64> {
        match self {
            Equation::ODE(_, lag, _, _, _) => (lag)(&V::from_vec(spp.clone())),
            Equation::OdeSolvers(_, lag, _, _, _) => (lag)(&V::from_vec(spp.clone())),
            Equation::SDE(_, _, _, _, _, _) => unimplemented!("Not Implemented"),
            Equation::Analytical(_, _, lag, _, _, _) => (lag)(&V::from_vec(spp.clone())),
        }
    }
    #[inline(always)]
    fn get_fa(&self, spp: &Vec<f64>) -> HashMap<usize, f64> {
        match self {
            Equation::ODE(_, _, fa, _, _) => (fa)(&V::from_vec(spp.clone())),
            Equation::OdeSolvers(_, _, fa, _, _) => (fa)(&V::from_vec(spp.clone())),
            Equation::SDE(_, _, _, _, _, _) => unimplemented!("Not Implemented"),
            Equation::Analytical(_, _, _, fa, _, _) => (fa)(&V::from_vec(spp.clone())),
        }
    }
}

trait FromVec {
    fn from_vec(vec: Vec<f64>) -> Self;
}

impl FromVec for faer::Col<f64> {
    fn from_vec(vec: Vec<f64>) -> Self {
        faer::Col::from_fn(vec.len(), |i| vec[i])
    }
}

#[inline(always)]
pub fn get_first_state(init: &Init, support_point: &Vec<f64>, cov: &Covariates) -> V {
    (init)(&V::from_vec(support_point.clone()), 0.0, cov) //TODO: Time hardcoded
}
