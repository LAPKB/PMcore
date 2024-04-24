use crate::routines::data::Observation;

/// ObsPred holds an observation and its prediction
pub struct ObsPred {
    pub time: f64,
    pub observation: f64,
    pub prediction: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
}

pub trait ToObsPred {
    fn to_obs_pred(self, pred: f64) -> ObsPred;
}

impl ToObsPred for Observation {
    fn to_obs_pred(self, pred: f64) -> ObsPred {
        ObsPred {
            time: self.time,
            observation: self.value,
            prediction: pred,
            outeq: self.outeq,
            errorpoly: self.errorpoly,
        }
    }
}
