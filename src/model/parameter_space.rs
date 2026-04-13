use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSpace {
    pub items: Vec<ParameterSpec>,
}

impl ParameterSpace {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(mut self, item: ParameterSpec) -> Self {
        self.items.push(item);
        self
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ParameterSpec> {
        self.items.iter()
    }

    pub fn names(&self) -> Vec<String> {
        self.items.iter().map(|item| item.name.clone()).collect()
    }

    pub fn finite_ranges(&self) -> Result<Vec<(f64, f64)>> {
        self.items
            .iter()
            .map(|parameter| match parameter.domain {
                ParameterDomain::Bounded { lower, upper } => Ok((lower, upper)),
                ParameterDomain::Positive {
                    lower: Some(lower),
                    upper: Some(upper),
                }
                | ParameterDomain::Unbounded {
                    lower: Some(lower),
                    upper: Some(upper),
                } => Ok((lower, upper)),
                _ => bail!(
                    "nonparametric initialization requires finite lower/upper bounds for parameter '{}'",
                    parameter.name
                ),
            })
            .collect()
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&ParameterSpace> for ParameterSpace {
    fn from(parameter_space: &ParameterSpace) -> Self {
        parameter_space.clone()
    }
}

fn default_estimate() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSpec {
    pub name: String,
    pub domain: ParameterDomain,
    #[serde(default)]
    pub transform: ParameterTransform,
    #[serde(default)]
    pub initial: Option<f64>,
    #[serde(default = "default_estimate")]
    pub estimate: bool,
    #[serde(default)]
    pub variability: ParameterVariability,
}

impl ParameterSpec {
    pub fn bounded(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self {
            name: name.into(),
            domain: ParameterDomain::Bounded { lower, upper },
            transform: ParameterTransform::Identity,
            initial: None,
            estimate: true,
            variability: ParameterVariability::Subject,
        }
    }

    pub fn positive(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            domain: ParameterDomain::Positive {
                lower: Some(0.0),
                upper: None,
            },
            transform: ParameterTransform::LogNormal,
            initial: None,
            estimate: true,
            variability: ParameterVariability::Subject,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterDomain {
    Positive {
        lower: Option<f64>,
        upper: Option<f64>,
    },
    Unbounded {
        lower: Option<f64>,
        upper: Option<f64>,
    },
    Bounded {
        lower: f64,
        upper: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ParameterTransform {
    #[default]
    Identity,
    LogNormal,
    Probit,
    Logit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ParameterVariability {
    FixedOnly,
    #[default]
    Subject,
    Occasion,
    SubjectAndOccasion,
}
