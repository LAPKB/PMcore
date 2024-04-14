use crate::routines::data::structures::*;
use std::collections::HashMap;

pub trait DataTrait {
    fn get_subjects(&self) -> Vec<&impl SubjectTrait>;
    /// Returns the number of subjects in the dataset
    fn nsubjects(&self) -> usize;
    /// Returns the number of observations in the dataset
    fn nobs(&self) -> usize;
}

/// [Subject] is a trait that represents a single individual in a dataset
pub trait SubjectTrait {
    fn get_occasions(&self) -> Vec<&impl OccasionTrait>;
    fn get_id(&self) -> &String;
}

/// Each [Subject] can have multiple occasions
pub trait OccasionTrait {
    fn get_events(
        &mut self,
        lagtime: Option<HashMap<usize, f64>>,
        bioavailability: Option<HashMap<usize, f64>>,
    ) -> Vec<&Event>;
    fn get_covariates(&self) -> Option<&impl CovariatesTrait>;
}

pub trait CovariatesTrait {
    fn get_covariate(&self, name: &str) -> Option<&impl CovariateInterpolator>;
}

/// Any [CovariateSegment] has to implement the [CovariateInterpolator] trait
pub trait CovariateInterpolator {
    fn interpolate(&self, time: f64) -> Option<f64>;
    fn in_interval(&self, time: f64) -> bool;
    fn description(&self) -> String;
}
