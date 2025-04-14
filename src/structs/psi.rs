use faer::Mat;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::{Array2, ArrayView2};
use pharmsol::prelude::simulator::psi;
use pharmsol::Data;
use pharmsol::Equation;
use pharmsol::ErrorModel;

use super::theta::Theta;

/// [Psi] is a structure that holds the likelihood for each subject (row), for each support point (column)
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

    /// Modify the [Psi::matrix] to only include the columns specified by `indices`
    pub(crate) fn filter_column_indices(&mut self, indices: &[usize]) {
        let matrix = self.matrix.to_owned();

        let new = Mat::from_fn(matrix.nrows(), indices.len(), |r, c| {
            *matrix.get(r, indices[c])
        });

        self.matrix = new;
    }

    /// Write the matrix to a CSV file
    pub fn write(&self, path: &str) {
        let mut writer = csv::Writer::from_path(path).unwrap();
        for row in self.matrix.row_iter() {
            writer
                .write_record(row.iter().map(|x| x.to_string()))
                .unwrap();
        }
    }
}

impl Default for Psi {
    fn default() -> Self {
        Psi::new()
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
) -> Psi {
    let psi_ndarray = psi(
        equation,
        subjects,
        &theta.matrix().clone().as_ref().into_ndarray().to_owned(),
        error_model,
        progress,
        cache,
    );

    psi_ndarray.view().into()
}
