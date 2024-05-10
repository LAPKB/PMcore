use crate::{
    prelude::data::Observation,
    routines::evaluation::sigma::{ErrorModel, ErrorType},
};

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct SubjectPredictions {
    predictions: Vec<Prediction>,
    flat_predictions: Array1<f64>,
    flat_observations: Array1<f64>,
}
impl SubjectPredictions {
    pub fn get_predictions(&self) -> &Vec<Prediction> {
        &self.predictions
    }

    fn subject_likelihood(&self, error_model: &ErrorModel) -> f64 {
        //TODO: This sigma should not be calculated here, we should precalculate it and inject it into the struct
        let sigma: Array1<f64> = self
            .predictions
            .iter()
            .map(|p| error_model.estimate_sigma(p))
            .collect();

        normal_likelihood(&self.flat_predictions, &self.flat_observations, &sigma)
    }
}
pub fn normal_likelihood(
    predictions: &Array1<f64>,
    observations: &Array1<f64>,
    sigma: &Array1<f64>,
) -> f64 {
    const FRAC_1_SQRT_2PI: f64 =
        std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;
    let diff = (observations - predictions).mapv(|x| x.powi(2));
    let two_sigma_sq = (2.0 * sigma).mapv(|x| x.powi(2));
    let aux_vec = FRAC_1_SQRT_2PI * (-&diff / two_sigma_sq).mapv(|x| x.exp()) / sigma;
    aux_vec.product()
}
impl From<Vec<Prediction>> for SubjectPredictions {
    fn from(predictions: Vec<Prediction>) -> Self {
        Self {
            flat_predictions: predictions.iter().map(|p| p.prediction).collect(),
            flat_observations: predictions.iter().map(|p| p.observation).collect(),
            predictions: predictions,
        }
    }
}

pub struct PopulationPredictions {
    pub subject_predictions: Array2<SubjectPredictions>,
}

impl Default for PopulationPredictions {
    fn default() -> Self {
        Self {
            subject_predictions: Array2::default((0, 0)),
        }
    }
}

impl PopulationPredictions {
    pub fn get_psi(&self, ep: &ErrorModel) -> Array2<f64> {
        let mut psi = Array2::zeros((
            self.subject_predictions.nrows(),
            self.subject_predictions.ncols(),
        ));
        psi.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                row.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(j, mut element)| {
                        element.fill(
                            self.subject_predictions
                                .get((i, j))
                                .unwrap()
                                .subject_likelihood(ep),
                        );
                    })
            });
        psi
    }
    pub fn get_predictions(&self) -> Vec<&Prediction> {
        self.subject_predictions
            .iter()
            .map(|sp| sp.get_predictions())
            .flatten()
            .collect()
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
    pub id: String,
    pub occasion: usize,
    pub time: f64,
    pub observation: f64,
    pub prediction: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
}

pub trait ToPrediction {
    fn to_obs_pred(&self, pred: f64, id: String, occasion: usize) -> Prediction;
}

impl ToPrediction for Observation {
    fn to_obs_pred(&self, pred: f64, id: String, occasion: usize) -> Prediction {
        Prediction {
            id: id,
            occasion: occasion,
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
            id: String::new(),
            occasion: 99,
            time: 0.0,
            observation: -99.0,
            prediction: -99.0,
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

trait SigmaEstimator {
    fn estimate_sigma(&self, prediction: &Prediction) -> f64;
}

impl<'a> SigmaEstimator for ErrorModel<'_> {
    fn estimate_sigma(&self, prediction: &Prediction) -> f64 {
        let (c0, c1, c2, c3) = match prediction.errorpoly {
            Some((c0, c1, c2, c3)) => (c0, c1, c2, c3),
            None => (self.c.0, self.c.1, self.c.2, self.c.3),
        };
        let alpha = c0
            + c1 * prediction.observation
            + c2 * prediction.observation.powi(2)
            + c3 * prediction.observation.powi(3);

        let res = match self.e_type {
            ErrorType::Add => (alpha.powi(2) + self.gl.powi(2)).sqrt(),
            ErrorType::Prop => self.gl * alpha,
        };

        if res.is_nan() || res < 0.0 {
            tracing::error!(
                "The computed standard deviation is either NaN or negative (SD = {}), coercing to 0",
                res
            );
            0.0
        } else {
            res
        }
    }
}
