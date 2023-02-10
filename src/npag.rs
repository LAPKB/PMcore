use crate::prelude::*;

const THETA_E: f64 = 1e-4; //convergence Criteria

pub fn npag<S>(sim_eng: Engine<S>, ranges: Vec<(f32,f32)>, settings_path: String, seed: u32)
where
    S: Simulate
{
    let settings = settings::read(settings_path);
    let theta0 = lds::sobol(1024, ranges, seed);
    let scenarios = datafile::parse(settings.paths.data);

    let mut eps  = 0.2;

    while eps > THETA_E {
        
        
    }
}