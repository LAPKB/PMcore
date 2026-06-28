use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
}
