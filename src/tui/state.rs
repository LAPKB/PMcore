#[derive(Debug)]
pub struct AppState{
    pub cycle: usize,
    pub objf: f64
}
impl AppState{
    pub fn new()->Self{
        Self{
            cycle: 0,
            objf: f64::INFINITY
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

