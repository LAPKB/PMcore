use crate::base::datafile::Scenario;
use interp::interp_slice;
pub trait Simulate{
    fn simulate(&self, params: Vec<f64>, y0: Vec<f64>, tspan:[f64;2], scenario: &Scenario) -> (Vec<f64>, Vec<f64>);
}

pub struct Engine<S>
where 
    S: Simulate
{
    sim: S
}

impl<S> Engine<S>
where
    S: Simulate
{
    pub fn new(sim: S) -> Self{
        Self{
            sim
        }
    }
    pub fn sim_obs(&self, scenario: &Scenario) -> Vec<f64>{
        let (x_out, y_out) = self.sim.simulate(
            vec![0.1,2.0],
            vec![0.0,0.0],
            [scenario.time.first().unwrap().clone() as f64, scenario.time.last().unwrap().clone() as f64],
            scenario
        );

        let y_intrp = interp_slice(&x_out, &y_out, &scenario.time_obs[..]);
        y_intrp
    }
}


