use crate::prelude::Data;

#[derive(Debug, Clone)]
pub struct AppState {
    pub cycle: usize,
    pub objf: f64,
    pub delta_objf: f64,
    pub nspp: usize,
    pub stop_text: String,
    pub gamlam: f64,
    pub options: Option<Data>,
}
impl AppState {
    pub fn new() -> Self {
        Self {
            cycle: 0,
            objf: f64::INFINITY,
            delta_objf: 0.0,
            nspp: 0,
            stop_text: "".to_string(),
            gamlam: 0.0,
            options: None,
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AppHistory {
    pub cycles: Vec<AppState>,
}

impl AppHistory {
    pub fn new() -> Self {
        AppHistory { cycles: Vec::new() }
    }

    pub fn add_cycle(&mut self, cycle: AppState) {
        self.cycles.push(cycle);
    }
}
