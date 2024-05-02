use crate::routines::data::Observation;

/// ObsPred holds an observation and its prediction
#[derive(Debug)]
pub struct ObsPred {
    pub time: f64,
    pub observation: f64,
    pub prediction: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
}

pub trait ToObsPred {
    fn to_obs_pred(&self, pred: f64) -> ObsPred;
}

impl ToObsPred for Observation {
    fn to_obs_pred(&self, pred: f64) -> ObsPred {
        ObsPred {
            time: self.time,
            observation: self.value,
            prediction: pred,
            outeq: self.outeq,
            errorpoly: self.errorpoly,
        }
    }
}

// Implement display for Obspred
impl std::fmt::Display for ObsPred {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.15}\tOuteq: {:.2}",
            self.time, self.observation, self.prediction, self.outeq
        )
    }
}
