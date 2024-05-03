use crate::routines::data::Observation;

/// IndObsPred holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct IndObsPred {
    pub time: f64,
    pub observation: f64,
    pub prediction: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
}

pub trait ToIndObsPred {
    fn to_obs_pred(&self, pred: f64) -> IndObsPred;
}

impl ToIndObsPred for Observation {
    fn to_obs_pred(&self, pred: f64) -> IndObsPred {
        IndObsPred {
            time: self.time,
            observation: self.value,
            prediction: pred,
            outeq: self.outeq,
            errorpoly: self.errorpoly,
        }
    }
}

impl Default for IndObsPred {
    fn default() -> Self {
        Self {
            time: 0.0,
            observation: 0.0,
            prediction: 0.0,
            outeq: 0,
            errorpoly: None,
        }
    }
}

// Implement display for IndObsPred
impl std::fmt::Display for IndObsPred {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.15}\tOuteq: {:.2}",
            self.time, self.observation, self.prediction, self.outeq
        )
    }
}
