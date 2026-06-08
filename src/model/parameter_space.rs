use serde::{Deserialize, Serialize};

/// Ordered collection of parameters.
///
/// Use `ParameterSpace<BoundedParameter>` for non-parametric problems and
/// `ParameterSpace<UnboundedParameter>` for parametric problems.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ParameterSpace<T> {
    pub items: Vec<T>,
}

impl<T> ParameterSpace<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: impl Into<T>) {
        self.items.push(item.into());
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, item: impl Into<T>) -> Self {
        self.push(item);
        self
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }
}

impl<T: ParameterMeta> ParameterSpace<T> {
    pub fn names(&self) -> Vec<String> {
        self.items.iter().map(|p| p.name().to_string()).collect()
    }
}

/// Helpers for bounded parameter spaces.
impl ParameterSpace<BoundedParameter> {
    /// Returns `(lower, upper)` for each parameter.
    pub fn finite_ranges(&self) -> Vec<(f64, f64)> {
        self.items.iter().map(|p| (p.lower, p.upper)).collect()
    }
}

impl<T> FromIterator<T> for ParameterSpace<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            items: iter.into_iter().collect(),
        }
    }
}

/// Common metadata exposed by parameter types.
pub trait ParameterMeta {
    fn name(&self) -> &str;
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

impl ParameterMeta for BoundedParameter {
    fn name(&self) -> &str {
        &self.name
    }
}

/// Converts a bounded parameter into a parametric parameter with logit scaling.
impl From<BoundedParameter> for UnboundedParameter {
    fn from(p: BoundedParameter) -> Self {
        UnboundedParameter {
            name: p.name,
            scale: ParameterScale::Logit {
                lower: p.lower,
                upper: p.upper,
            },
            initial: None,
            estimate: true,
        }
    }
}

/// Parametric parameter with an optional scale transform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnboundedParameter {
    pub name: String,
    pub scale: ParameterScale,
    pub initial: Option<f64>,
    pub estimate: bool,
}

impl UnboundedParameter {
    /// Creates a parameter with an explicit scale.
    pub fn new(name: impl Into<String>, scale: ParameterScale) -> Self {
        Self {
            name: name.into(),
            scale,
            initial: None,
            estimate: true,
        }
    }

    /// Creates a parameter on identity scale.
    pub fn real(name: impl Into<String>) -> Self {
        Self::new(name, ParameterScale::Identity)
    }

    /// Sets an initial value.
    pub fn with_initial(mut self, value: f64) -> Self {
        self.initial = Some(value);
        self
    }
}

impl ParameterMeta for UnboundedParameter {
    fn name(&self) -> &str {
        &self.name
    }
}

/// Scale transform for parametric parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ParameterScale {
    /// Identity transform.
    Identity,
    /// Log transform.
    Log,
    /// Logistic transform on `(lower, upper)`.
    Logit { lower: f64, upper: f64 },
    /// Probit transform on `(lower, upper)`.
    Probit { lower: f64, upper: f64 },
}

impl Default for ParameterScale {
    fn default() -> Self {
        ParameterScale::Identity
    }
}

/// Entry point for building parameter declarations.
///
/// ```ignore
/// use pmcore::prelude::*;
///
/// // Non-parametric: only bounded parameters are accepted.
/// builder.parameter(Parameter::bounded("ke", 0.001, 3.0));
///
/// // Parametric: pick the scale explicitly.
/// builder.parameter(Parameter::log("ke"));
/// builder.parameter(Parameter::logit("frac", 0.0, 1.0));
/// builder.parameter(Parameter::bounded("v", 25.0, 250.0)); // mapped to Logit
/// ```
pub struct Parameter;

impl Parameter {
    /// Creates a bounded parameter.
    pub fn bounded(name: impl Into<String>, lower: f64, upper: f64) -> BoundedParameter {
        BoundedParameter::new(name, lower, upper)
    }

    /// Creates a parametric parameter on identity scale.
    pub fn real(name: impl Into<String>) -> UnboundedParameter {
        UnboundedParameter::real(name)
    }

    /// Creates a parametric parameter with an explicit scale.
    pub fn scaled(name: impl Into<String>, scale: ParameterScale) -> UnboundedParameter {
        UnboundedParameter::new(name, scale)
    }

    /// Creates a parametric parameter on log scale.
    pub fn log(name: impl Into<String>) -> UnboundedParameter {
        UnboundedParameter::new(name, ParameterScale::Log)
    }

    /// Creates a parametric parameter on logit scale.
    pub fn logit(name: impl Into<String>, lower: f64, upper: f64) -> UnboundedParameter {
        UnboundedParameter::new(name, ParameterScale::Logit { lower, upper })
    }

    /// Creates a parametric parameter on probit scale.
    pub fn probit(name: impl Into<String>, lower: f64, upper: f64) -> UnboundedParameter {
        UnboundedParameter::new(name, ParameterScale::Probit { lower, upper })
    }
}
