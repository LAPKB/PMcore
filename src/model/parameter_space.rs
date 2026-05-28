use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterSpace {
    NonParametric(NonParametricParameters),
    Parametric(ParametricParameters),
}

impl ParameterSpace {
    /// Returns the total number of parameters across either variant
    pub fn len(&self) -> usize {
        match self {
            Self::NonParametric(p) => p.len(),
            Self::Parametric(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves the names of the parameters, regardless of the framework
    pub fn names(&self) -> Vec<String> {
        match self {
            Self::NonParametric(p) => p.names(),
            Self::Parametric(p) => p.names(),
        }
    }

    /// Convenience accessor for the non-parametric inner struct
    pub fn as_nonparametric(&self) -> Option<&NonParametricParameters> {
        match self {
            Self::NonParametric(p) => Some(p),
            Self::Parametric(_) => None,
        }
    }

    /// Convenience accessor for the parametric inner struct
    pub fn as_parametric(&self) -> Option<&ParametricParameters> {
        match self {
            Self::Parametric(p) => Some(p),
            Self::NonParametric(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NonParametricParameters {
    pub items: Vec<BoundedParameter>,
}

impl NonParametricParameters {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: BoundedParameter) {
        self.items.push(item);
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, item: BoundedParameter) -> Self {
        self.push(item);
        self
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, BoundedParameter> {
        self.items.iter()
    }

    pub fn names(&self) -> Vec<String> {
        self.items.iter().map(|item| item.name.clone()).collect()
    }

    /// Infallible! The type system guarantees lower and upper bounds exist.
    pub fn finite_ranges(&self) -> Vec<(f64, f64)> {
        self.items
            .iter()
            .map(|parameter| (parameter.lower, parameter.upper))
            .collect()
    }
}

impl Default for NonParametricParameters {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoundedParameter {
    pub name: String,
    pub lower: f64,
    pub upper: f64,
}

impl BoundedParameter {
    pub fn new(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self {
            name: name.into(),
            lower,
            upper,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParametricParameters {
    pub items: Vec<UnboundedParameter>,
}

impl ParametricParameters {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: UnboundedParameter) {
        self.items.push(item);
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, item: UnboundedParameter) -> Self {
        self.push(item);
        self
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, UnboundedParameter> {
        self.items.iter()
    }

    pub fn names(&self) -> Vec<String> {
        self.items.iter().map(|item| item.name.clone()).collect()
    }
}

impl Default for ParametricParameters {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnboundedParameter {
    pub name: String,
    pub domain: ParameterDomain,
    pub transform: ParameterTransform,
    pub initial: Option<f64>,
    pub estimate: bool,
}

impl UnboundedParameter {
    pub fn bounded(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self {
            name: name.into(),
            domain: ParameterDomain::Bounded { lower, upper },
            transform: ParameterTransform::Identity,
            initial: None,
            estimate: true,
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
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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
