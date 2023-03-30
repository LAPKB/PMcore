use crate::base::datafile::Scenario;

///
/// return the predicted values for the given scenario and parameters
/// where the second element of the tuple is the predicted values
/// one per observation time in scenario and in the same order
/// it is not relevant the outeq of the specific event.
pub trait Simulate {
    fn simulate(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64>;
}

pub struct Engine<S>
where
    S: Simulate,
{
    sim: S,
}

impl<S> Engine<S>
where
    S: Simulate,
{
    pub fn new(sim: S) -> Self {
        Self { sim }
    }
    pub fn pred(&self, scenario: &Scenario, params: Vec<f64>) -> Vec<f64> {
        self.sim.simulate(params, scenario)
    }
}
