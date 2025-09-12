use faer::Col;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// The weight (probabilities) for each support point in the model.
///
/// This struct is used to hold the weights for each support point in the model.
/// It is a thin wrapper around [faer::Col<f64>] to provide additional functionality and context
#[derive(Debug, Clone)]
pub struct Weights {
    weights: Col<f64>,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            weights: Col::from_fn(0, |_| 0.0),
        }
    }
}

impl Weights {
    pub fn new(weights: Col<f64>) -> Self {
        Self { weights }
    }

    /// Create a new [Weights] instance from a vector of weights.
    pub fn from_vec(weights: Vec<f64>) -> Self {
        Self {
            weights: Col::from_fn(weights.len(), |i| weights[i]),
        }
    }

    /// Get a reference to the weights.
    pub fn weights(&self) -> &Col<f64> {
        &self.weights
    }

    /// Get a mutable reference to the weights.
    pub fn weights_mut(&mut self) -> &mut Col<f64> {
        &mut self.weights
    }

    /// Get the number of weights.
    pub fn len(&self) -> usize {
        self.weights.nrows()
    }

    /// Get a vector representation of the weights.
    pub fn to_vec(&self) -> Vec<f64> {
        self.weights.iter().cloned().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.weights.iter().cloned()
    }
}

impl Serialize for Weights {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_vec().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Weights {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let weights_vec = Vec::<f64>::deserialize(deserializer)?;
        Ok(Self::from_vec(weights_vec))
    }
}

impl From<Vec<f64>> for Weights {
    fn from(weights: Vec<f64>) -> Self {
        Self::from_vec(weights)
    }
}

impl From<Col<f64>> for Weights {
    fn from(weights: Col<f64>) -> Self {
        Self { weights }
    }
}

impl Index<usize> for Weights {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.weights[index]
    }
}

impl IndexMut<usize> for Weights {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.weights[index]
    }
}
