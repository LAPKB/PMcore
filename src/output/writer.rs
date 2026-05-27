use pharmsol::Censor;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
struct SharedPredictionRow {
    id: String,
    time: f64,
    outeq: usize,
    block: usize,
    obs: Option<f64>,
    cens: Censor,
    pred_population: f64,
    pred_individual: f64,
    residual_population: Option<f64>,
    residual_individual: Option<f64>,
    source_method: String,
}
