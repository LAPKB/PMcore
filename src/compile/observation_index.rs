use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ObservationIndex {
    pub records: Vec<ObservationRecord>,
}

impl ObservationIndex {
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservationRecord {
    pub subject_index: usize,
    pub occasion_index: usize,
    pub event_index: usize,
    pub outeq: usize,
    pub time: f64,
}
