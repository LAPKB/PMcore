use crate::base::datafile::Scenario;
use interp::interp_slice;
use std::{collections::HashSet, iter};

pub trait Simulate {
    fn simulate(
        &self,
        params: Vec<f64>,
        tspan: [f64; 2],
        scenario: &Scenario,
    ) -> (Vec<f64>, Vec<Vec<f64>>);
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
        let (x_out, y_out) = self.sim.simulate(
            params,
            [
                *scenario.time.first().unwrap(),
                *scenario.time.last().unwrap(),
            ],
            scenario,
        );
        let mut y_intrp: Vec<Vec<f64>> = vec![];
        for (i, out) in y_out.iter().enumerate() {
            y_intrp.push(interp_slice(
                &x_out,
                out,
                &scenario.time_obs.get(i).unwrap()[..],
            ));
        }
        y_intrp.into_iter().flatten().collect::<Vec<f64>>()
    }

    pub fn pred_full(&self, scenario: &Scenario, params: Vec<f64>, dt: f64) -> Vec<(f64, f64)> {
        let start_time = *scenario.time.first().unwrap();
        let end_time = *scenario.time.last().unwrap();

        // Generate custom_times based on the given interval (dt)
        let predicted_times: Vec<f64> = iter::successors(Some(start_time), |t| Some(t + dt))
            .take_while(|t| *t <= end_time)
            .collect();

        // Concatenate observed times with predicted times
        let custom_times: Vec<f64> = scenario
            .time_flat
            .iter()
            .chain(predicted_times.iter())
            .cloned()
            .collect();

        let (x_out, y_out) = self.sim.simulate(params, [start_time, end_time], scenario);

        let y_intrp: Vec<Vec<f64>> = y_out
            .iter()
            .map(|out| interp_slice(&x_out, out, &custom_times))
            .collect();

        let predictions: Vec<(f64, f64)> = custom_times
            .into_iter()
            .zip(y_intrp.into_iter().flatten())
            .collect();

        // Filter out duplicate rows by time
        let mut unique_predictions = Vec::new();
        let mut seen_times = HashSet::new();
        for (time, value) in predictions {
            let time_int = (time * 1000000.0) as i64;
            if seen_times.insert(time_int) {
                unique_predictions.push((time, value));
            }
        }

        // Sort the predictions by time (ascending)
        unique_predictions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        unique_predictions
    }
}
