use crate::prelude::output::NPCycle;

#[derive(Debug, Clone)]
// pub struct AppState {
//     pub cycle: usize,
//     pub objf: f64,
//     pub delta_objf: f64,
//     pub nspp: usize,
//     pub stop_text: String,
//     pub gamlam: f64,
// }
// impl AppState {
//     pub fn new() -> Self {
//         Self {
//             cycle: 0,
//             objf: 0.0,
//             delta_objf: 0.0,
//             nspp: 0,
//             stop_text: "".to_string(),
//             gamlam: 0.0,
//         }
//     }
// }

// impl Default for AppState {
//     fn default() -> Self {
//         Self::new()
//     }
// }

pub struct AppHistory {
    pub cycles: Vec<NPCycle>,
}

impl AppHistory {
    pub fn new() -> Self {
        AppHistory { cycles: Vec::new() }
    }

    pub fn add_cycle(&mut self, cycle: NPCycle) {
        self.cycles.push(cycle);
    }
}
impl Default for AppHistory {
    fn default() -> Self {
        Self::new()
    }
}
