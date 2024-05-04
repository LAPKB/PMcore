use ndarray::Array2;

use crate::routines::{data::Observation, evaluation::sigma::ErrorPoly};

#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
}
impl SubjectPredictions {
    pub fn get_predictions(&self) -> &Vec<Prediction> {
        &self.predictions
    }
}
impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            predictions: predictions,
        }
    }
}

pub struct PopulationPredictions {
    subject_predictions: Array2<SubjectPredictions>,
}

impl PopulationPredictions {
    pub fn get_psi(&self, ep: &ErrorPoly) -> Array2<f64> {
        unimplemented!()
    }
}

impl From<Array2<SubjectPredictions>> for PopulationPredictions {
    fn from(subject_predictions: Array2<SubjectPredictions>) -> Self {
        Self {
            subject_predictions: subject_predictions,
        }
    }
}

/// Prediction holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct Prediction {
    pub time: f64,
    pub observation: f64,
    pub prediction: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
}

impl Prediction {
    pub fn likelihood(&self, ep: &ErrorPoly) -> f64 {
        unimplemented!()
    }
}

pub trait ToPrediction {
    fn to_obs_pred(&self, pred: f64) -> Prediction;
}

impl ToPrediction for Observation {
    fn to_obs_pred(&self, pred: f64) -> Prediction {
        Prediction {
            time: self.time,
            observation: self.value,
            prediction: pred,
            outeq: self.outeq,
            errorpoly: self.errorpoly,
        }
    }
}

impl Default for Prediction {
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

// Implement display for Prediction
impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.15}\tOuteq: {:.2}",
            self.time, self.observation, self.prediction, self.outeq
        )
    }
}
