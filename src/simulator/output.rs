use crate::routines::data::ObsError;
use crate::routines::data::Observation;

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

/// ObsPred holds an observation and its prediction
#[derive(Debug, Clone)]
pub struct ObsPred {
    pub time: f64,
    pub observation: f64,
    pub prediction: f64,
    pub outeq: usize,
    pub obserror: ObsError,
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
            obserror: self.obserror,
        }
    }
}

// Implement display for IndObsPred
impl std::fmt::Display for ObsPred {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Time: {:.2}\tObs: {:.4}\tPred: {:.15}\tOuteq: {:.2}",
            self.time, self.observation, self.prediction, self.outeq
        )
    }
}

impl ObsPred {
    pub fn likelihood(&self, gamlam: f64) -> f64 {
        let diff: f64 = (self.observation - self.prediction).powi(2);
        let sigma: f64 = self.obserror.sigma(self.observation, gamlam);
        let two_sigma_sq: f64 = (2.0 * sigma).powi(2);
        let likelihood: f64 = FRAC_1_SQRT_2PI * (-&diff / two_sigma_sq).exp() / sigma;
        likelihood
    }

    pub fn update_errorpoly(&mut self, errorpoly: &(f64, f64, f64, f64)) {
        self.obserror.update_errorpoly(errorpoly);
    }
}

pub struct OccasionOutput {
    pub id: String,
    pub occasion: usize,
    pub obspred: Vec<ObsPred>,
}

pub struct SubjectOutput {
    pub id: String,
    pub occasionoutput: Vec<OccasionOutput>,
}

impl SubjectOutput {
    pub fn likelihood(&self, gamlam: &f64) -> f64 {
        self.occasionoutput
            .iter()
            .map(|occasion| {
                occasion
                    .obspred
                    .iter()
                    .map(|obspred| obspred.likelihood(*gamlam))
                    .sum::<f64>()
            })
            .product::<f64>()
    }
}
