use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct AppState {
    pub cycle: usize,
    pub objf: f64,
    pub delta_objf : f64,
    pub theta: Array2<f64>,
    pub stop_text : String,
}
impl AppState {
    pub fn new() -> Self {
        Self {
            cycle: 0,
            objf: f64::INFINITY,
            delta_objf: 0.0,
            theta: Array2::default((0, 0)),
            stop_text : "".to_string(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
