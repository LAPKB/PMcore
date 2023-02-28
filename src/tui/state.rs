use ndarray::Array2;

#[derive(Debug)]
pub struct AppState{
    pub cycle: usize,
    pub objf: f64,
    pub theta: Array2<f64>
}
impl AppState{
    pub fn new()->Self{
        Self{
            cycle: 0,
            objf: f64::INFINITY,
            theta: Array2::default((0,0))
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

