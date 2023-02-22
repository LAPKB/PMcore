use crate::base::datafile::Scenario;
use interp::interp_slice;
pub trait Simulate{
    fn simulate(&self, params: Vec<f64>, tspan:[f64;2], scenario: &Scenario) -> (Vec<f64>, Vec<Vec<f64>>);
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
    pub fn pred(&self, scenario: &Scenario, params: Vec<f64>) -> Vec<f64>{
        let (x_out, y_out) = self.sim.simulate(
            params,
            [*scenario.time.first().unwrap(), *scenario.time.last().unwrap()],
            scenario
        );
        let mut y_intrp: Vec<Vec<f64>> = vec![]; 
        for (i,out) in y_out.iter().enumerate(){
            y_intrp.push(interp_slice(&x_out, out, &scenario.time_obs.get(i).unwrap()[..]));
        }
        y_intrp.into_iter().flatten().collect::<Vec<f64>>()
    }


}
