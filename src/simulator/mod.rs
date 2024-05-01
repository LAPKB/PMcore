pub mod analytical;
pub mod likelihood;
pub mod ode;
pub mod ode_solvers;
use crate::{
    prelude::data::Event,
    routines::{
        data::{Covariates, Infusion, OccasionTrait, SubjectTrait},
        datafile::Scenario,
    },
    simulator::likelihood::{ObsPred, ToObsPred},
};

pub type T = f64;
// pub type V = faer::Col<T>;
// pub type M = faer::Mat<T>;
pub type V = nalgebra::DVector<T>;
pub type M = nalgebra::DMatrix<T>;

pub type DiffEq = fn(&V, &V, T, &mut V, V, &Covariates);
pub type Init = fn(&V, T, &Covariates) -> V;
pub type Out = fn(&V, &V, T, &Covariates) -> V;
pub type AnalyticalEq = fn(&V, &V, T, V, &Covariates) -> V;
pub type SecEq = fn(&mut V, &Covariates);

pub enum Equation {
    OdeSolvers(DiffEq, Init, Out),
    ODE(DiffEq, Init, Out),
    SDE(DiffEq, DiffEq, Init, Out),
    Analytical(AnalyticalEq, SecEq, Init, Out),
}

impl Equation {
    pub fn new_ode(diffeq: DiffEq, init: Init, out: Out) -> Self {
        Equation::ODE(diffeq, init, out)
    }
    pub fn new_analytical(eq: AnalyticalEq, seq_eq: SecEq, init: Init, out: Out) -> Self {
        Equation::Analytical(eq, seq_eq, init, out)
    }

    pub fn new_ode_solvers(diffeq: DiffEq, init: Init, out: Out) -> Self {
        Equation::OdeSolvers(diffeq, init, out)
    }

    pub fn simulate_scenario(&self, scenario: &Scenario, support_point: &Vec<f64>) -> Vec<ObsPred> {
        let init = self.get_init();
        let out = self.get_out();
        let covariates = Covariates::new();
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
                        x[event.input.unwrap() - 1] += event.dose.unwrap();
                    }
                } else if event.evid == 0 {
                    //obs
                    let pred = *(out)(
                        &x,
                        &V::from_vec(support_point.clone()),
                        event.time,
                        &covariates,
                    )
                    .get(event.outeq.unwrap() - 1)
                    .unwrap();
                    yout.push(ObsPred {
                        time: event.time,
                        observation: event.out.unwrap(),
                        prediction: pred,
                        outeq: event.outeq.unwrap(),
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
        yout
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
            let covariates = occasion.get_covariates().unwrap();
            let mut x = get_first_state(init, support_point, &covariates);
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
                        )[observation.outeq];

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
            Equation::ODE(eqn, _init, _out) => ode::simulate_ode_event(
                eqn,
                x,
                support_point,
                covariates,
                infusions,
                start_time,
                end_time,
            ),
            Equation::OdeSolvers(eqn, _init, _out) => ode_solvers::simulate_ode_event(
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
            Equation::Analytical(eq, seq_eq, _, _) => analytical::simulate_analytical_event(
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
    fn get_init(&self) -> &Init {
        match self {
            Equation::ODE(_, init, _) => init,
            Equation::OdeSolvers(_, init, _) => init,
            Equation::SDE(_, _, init, _) => init,
            Equation::Analytical(_, _, init, _) => init,
        }
    }
    fn get_out(&self) -> &Out {
        match self {
            Equation::ODE(_, _, out) => out,
            Equation::OdeSolvers(_, _, out) => out,
            Equation::SDE(_, _, _, out) => out,
            Equation::Analytical(_, _, _, out) => out,
        }
    }
}

#[inline(always)]
pub fn get_first_state(init: &Init, support_point: &Vec<f64>, cov: &Covariates) -> V {
    (init)(&V::from_vec(support_point.clone()), 0.0, cov) //TODO: Time hardcoded
}
