use crate::prelude::output::NPCycle;

#[derive(Debug, Clone)]
pub struct CycleHistory {
    pub cycles: Vec<NPCycle>,
}

impl CycleHistory {
    pub fn new() -> Self {
        CycleHistory { cycles: Vec::new() }
    }

    pub fn add_cycle(&mut self, cycle: NPCycle) {
        self.cycles.push(cycle);
    }
}
impl Default for CycleHistory {
    fn default() -> Self {
        Self::new()
    }
}
