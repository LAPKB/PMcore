use faer::Mat;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::{Array2, ArrayView2};
use pharmsol::prelude::simulator::psi;
use pharmsol::Data;
use pharmsol::Equation;
use pharmsol::ErrorModel;

use super::theta::Theta;

#[derive(Debug, Clone, PartialEq)]
pub struct Psi {
    matrix: Mat<f64>,
}

impl Psi {
    pub fn new() -> Self {
        Psi { matrix: Mat::new() }
    }

    pub fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    pub fn nspp(&self) -> usize {
        self.matrix.nrows()
    }

    pub fn nsub(&self) -> usize {
        self.matrix.ncols()
    }
}

impl From<Array2<f64>> for Psi {
    fn from(array: Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Psi { matrix }
    }
}

impl From<Mat<f64>> for Psi {
    fn from(matrix: Mat<f64>) -> Self {
        Psi { matrix }
    }
}

impl From<ArrayView2<'_, f64>> for Psi {
    fn from(array_view: ArrayView2<'_, f64>) -> Self {
        let matrix = array_view.into_faer().to_owned();
        Psi { matrix }
    }
}

impl From<&Array2<f64>> for Psi {
    fn from(array: &Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Psi { matrix }
    }
}

pub(crate) fn calculate_psi(
    equation: &impl Equation,
    subjects: &Data,
    theta: &Theta,
    error_model: &ErrorModel,
    progress: bool,
    cache: bool,
) -> Array2<f64> {
    let psi = psi(
        equation,
        subjects,
        &theta.matrix().clone().as_ref().into_ndarray().to_owned(),
        error_model,
        progress,
        cache,
    );

    psi
}
