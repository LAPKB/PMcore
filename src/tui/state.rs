use crate::prelude::output::NPCycle;

#[derive(Debug, Clone)]
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
