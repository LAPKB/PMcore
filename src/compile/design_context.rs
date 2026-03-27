use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DesignContext {
    pub parameter_names: Vec<String>,
    pub subjects: Vec<SubjectDesign>,
    pub occasions: Vec<OccasionDesign>,
    pub structured_covariates: StructuredCovariateDesign,
}

impl DesignContext {
    pub fn subject_count(&self) -> usize {
        self.subjects.len()
    }

    pub fn occasion_count(&self) -> usize {
        self.occasions.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubjectDesign {
    pub subject_index: usize,
    pub id: String,
    pub occasion_count: usize,
    pub observation_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OccasionDesign {
    pub subject_index: usize,
    pub occasion_index: usize,
    pub event_count: usize,
    pub observation_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StructuredCovariateDesign {
    pub subject_columns: Vec<String>,
    pub subject_rows: Vec<SubjectCovariateRow>,
    pub occasion_columns: Vec<String>,
    pub occasion_rows: Vec<OccasionCovariateRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubjectCovariateRow {
    pub subject_index: usize,
    pub id: String,
    pub anchor_time: f64,
    pub values: Vec<Option<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OccasionCovariateRow {
    pub subject_index: usize,
    pub occasion_index: usize,
    pub anchor_time: f64,
    pub values: Vec<Option<f64>>,
}