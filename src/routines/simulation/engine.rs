use crate::routines::datafile::CovLine;
use crate::routines::datafile::Infusion;
use crate::routines::datafile::Scenario;
use std::collections::HashMap;
#[derive(Clone, Debug)]
pub struct Engine<S>
where
    S: Predict<'static> + Clone,
{
    ode: S,
}

impl<S> Engine<S>
where
    S: Predict<'static> + Clone,
{
    pub fn new(ode: S) -> Self {
        Self { ode }
    }
    pub fn pred(&self, scenario: Scenario, params: Vec<f64>) -> Vec<f64> {
        let (mut system, scenario) = self.ode.initial_system(&params, scenario.clone());
        let mut yout = vec![];
        let mut x = self.ode.initial_state();
        let mut index: usize = 0;
        for block in scenario.blocks {
            self.ode.add_covs(&mut system, Some(block.covs));
            for event in &block.events {
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        self.ode.add_infusion(
                            &mut system,
                            Infusion {
                                time: event.time,
                                dur: event.dur.unwrap(),
                                amount: event.dose.unwrap(),
                                compartment: event.input.unwrap() - 1,
                            },
                        );
                    } else {
                        //dose
                        self.ode
                            .add_dose(&mut x, event.dose.unwrap(), event.input.unwrap() - 1);
                    }
                } else if event.evid == 0 {
                    //obs
                    yout.push(
                        self.ode
                            .get_output(event.time, &x, &system, event.outeq.unwrap()),
                    )
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    // TODO: use the last dx as the initial one for the next simulation.
                    self.ode.ode_step(&mut x, &system, event.time, *next_time);
                }
                index += 1;
            }
        }
        yout
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    params: HashMap<String, f64>,
    _scenario: Scenario,
    _infusions: Vec<Infusion>,
    _cov: Option<HashMap<String, CovLine>>,
}
impl Model {
    pub fn get_param(&self, str: &str) -> f64 {
        *self.params.get(str).unwrap()
    }
}

/// return the predicted values for the given scenario and parameters
/// where the second element of the tuple is the predicted values
/// one per observation time in scenario and in the same order
/// it is not relevant the outeq of the specific event.
pub trait Predict<'a> {
    type Model: 'a + Clone;
    type State;
    fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario);
    fn initial_state(&self) -> Self::State;
    fn add_covs(&self, system: &mut Self::Model, cov: Option<HashMap<String, CovLine>>);
    fn add_infusion(&self, system: &mut Self::Model, infusion: Infusion);
    fn add_dose(&self, state: &mut Self::State, dose: f64, compartment: usize);
    fn get_output(&self, time: f64, state: &Self::State, system: &Self::Model, outeq: usize)
        -> f64;
    fn ode_step(
        &self,
        _state: &mut Self::State,
        _system: &Self::Model,
        _time: f64,
        _next_time: f64,
    ) {
    }
}
