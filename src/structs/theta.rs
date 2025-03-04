use faer::Mat;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone, PartialEq)]
pub struct Theta {
    matrix: Mat<f64>,
}

impl Theta {
    pub fn new() -> Self {
        Theta { matrix: Mat::new() }
    }

    pub fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    pub fn matrix_ndarray(&self) -> Array2<f64> {
        self.matrix.clone().as_ref().into_ndarray().to_owned()
    }

    pub fn nspp(&self) -> usize {
        self.matrix.nrows()
    }
}

impl From<Array2<f64>> for Theta {
    fn from(array: Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Theta { matrix }
    }
}

impl From<Mat<f64>> for Theta {
    fn from(matrix: Mat<f64>) -> Self {
        Theta { matrix }
    }
}

impl From<ArrayView2<'_, f64>> for Theta {
    fn from(array_view: ArrayView2<'_, f64>) -> Self {
        let matrix = array_view.into_faer().to_owned();
        Theta { matrix }
    }
}

impl From<&Array2<f64>> for Theta {
    fn from(array: &Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Theta { matrix }
    }
}
