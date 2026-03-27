use faer::{Col, Mat};
use pharmsol::Equation;

use super::{FimMethod, ParametricWorkspace, UncertaintyEstimates};

pub fn estimates<E: Equation>(workspace: &ParametricWorkspace<E>) -> &UncertaintyEstimates {
    workspace.uncertainty()
}

pub fn has_fim<E: Equation>(workspace: &ParametricWorkspace<E>) -> bool {
    workspace.uncertainty().has_fim()
}

pub fn has_standard_errors<E: Equation>(workspace: &ParametricWorkspace<E>) -> bool {
    workspace.uncertainty().has_standard_errors()
}

pub fn se_mu<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Col<f64>> {
    workspace.uncertainty().se_mu.as_ref()
}

pub fn fim<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Mat<f64>> {
    workspace.uncertainty().fim.as_ref()
}

pub fn fim_method<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<FimMethod> {
    workspace.uncertainty().fim_method
}