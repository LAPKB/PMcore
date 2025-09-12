use anyhow::bail;
use anyhow::Result;
use faer::Mat;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::{Array2, ArrayView2};
use pharmsol::prelude::simulator::psi;
use pharmsol::Data;
use pharmsol::Equation;
use pharmsol::ErrorModels;
use serde::{Deserialize, Serialize};

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

    /// Write the psi matrix to a CSV writer
    /// Each row represents a subject, each column represents a support point
    pub fn to_csv<W: std::io::Write>(&self, writer: W) -> Result<()> {
        let mut csv_writer = csv::Writer::from_writer(writer);

        // Write each row
        for i in 0..self.matrix.nrows() {
            let row: Vec<f64> = (0..self.matrix.ncols())
                .map(|j| *self.matrix.get(i, j))
                .collect();
            csv_writer.serialize(row)?;
        }

        csv_writer.flush()?;
        Ok(())
    }

    /// Read psi matrix from a CSV reader
    /// Each row represents a subject, each column represents a support point
    pub fn from_csv<R: std::io::Read>(reader: R) -> Result<Self> {
        let mut csv_reader = csv::Reader::from_reader(reader);
        let mut rows: Vec<Vec<f64>> = Vec::new();

        for result in csv_reader.deserialize() {
            let row: Vec<f64> = result?;
            rows.push(row);
        }

        if rows.is_empty() {
            bail!("CSV file is empty");
        }

        let nrows = rows.len();
        let ncols = rows[0].len();

        // Verify all rows have the same length
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                bail!("Row {} has {} columns, expected {}", i, row.len(), ncols);
            }
        }

        // Create matrix from rows
        let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);

        Ok(Psi { matrix: mat })
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

impl Serialize for Psi {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(self.matrix.nrows()))?;

        // Serialize each row as a vector
        for i in 0..self.matrix.nrows() {
            let row: Vec<f64> = (0..self.matrix.ncols())
                .map(|j| *self.matrix.get(i, j))
                .collect();
            seq.serialize_element(&row)?;
        }

        seq.end()
    }
}

impl<'de> Deserialize<'de> for Psi {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{SeqAccess, Visitor};
        use std::fmt;

        struct PsiVisitor;

        impl<'de> Visitor<'de> for PsiVisitor {
            type Value = Psi;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of rows (vectors of f64)")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut rows: Vec<Vec<f64>> = Vec::new();

                while let Some(row) = seq.next_element::<Vec<f64>>()? {
                    rows.push(row);
                }

                if rows.is_empty() {
                    return Err(serde::de::Error::custom("Empty matrix not allowed"));
                }

                let nrows = rows.len();
                let ncols = rows[0].len();

                // Verify all rows have the same length
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != ncols {
                        return Err(serde::de::Error::custom(format!(
                            "Row {} has {} columns, expected {}",
                            i,
                            row.len(),
                            ncols
                        )));
                    }
                }

                // Create matrix from rows
                let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);

                Ok(Psi { matrix: mat })
            }
        }

        deserializer.deserialize_seq(PsiVisitor)
    }
}

pub(crate) fn calculate_psi(
    equation: &impl Equation,
    subjects: &Data,
    theta: &Theta,
    error_models: &ErrorModels,
    progress: bool,
    cache: bool,
) -> Result<Psi> {
    let psi_ndarray = psi(
        equation,
        subjects,
        &theta.matrix().clone().as_ref().into_ndarray().to_owned(),
        error_models,
        progress,
        cache,
    )?;

    Ok(psi_ndarray.view().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_from_array2() {
        // Create a test 2x3 array
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let psi = Psi::from(array.clone());

        // Check dimensions
        assert_eq!(psi.nspp(), 2);
        assert_eq!(psi.nsub(), 3);

        // Check values by converting back to ndarray and comparing
        let result_array = psi.matrix().as_ref().into_ndarray();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(result_array[[i, j]], array[[i, j]]);
            }
        }
    }

    #[test]
    fn test_from_array2_ref() {
        // Create a test 3x2 array
        let array =
            Array2::from_shape_vec((3, 2), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();

        let psi = Psi::from(&array);

        // Check dimensions
        assert_eq!(psi.nspp(), 3);
        assert_eq!(psi.nsub(), 2);

        // Check values by converting back to ndarray and comparing
        let result_array = psi.matrix().as_ref().into_ndarray();
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(result_array[[i, j]], array[[i, j]]);
            }
        }
    }

    #[test]
    fn test_nspp() {
        // Test with a 4x2 matrix
        let array =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nspp(), 4);
    }

    #[test]
    fn test_nspp_empty() {
        // Test with empty matrix
        let psi = Psi::new();
        assert_eq!(psi.nspp(), 0);
    }

    #[test]
    fn test_nspp_single_row() {
        // Test with 1x3 matrix
        let array = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nspp(), 1);
    }

    #[test]
    fn test_nsub() {
        // Test with a 2x5 matrix
        let array = Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nsub(), 5);
    }

    #[test]
    fn test_nsub_empty() {
        // Test with empty matrix
        let psi = Psi::new();
        assert_eq!(psi.nsub(), 0);
    }

    #[test]
    fn test_nsub_single_column() {
        // Test with 3x1 matrix
        let array = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nsub(), 1);
    }

    #[test]
    fn test_from_implementations_consistency() {
        // Test that both From implementations produce the same result
        let array = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();

        let psi_from_owned = Psi::from(array.clone());
        let psi_from_ref = Psi::from(&array);

        // Both should have the same dimensions
        assert_eq!(psi_from_owned.nspp(), psi_from_ref.nspp());
        assert_eq!(psi_from_owned.nsub(), psi_from_ref.nsub());

        // And the same values
        let owned_array = psi_from_owned.matrix().as_ref().into_ndarray();
        let ref_array = psi_from_ref.matrix().as_ref().into_ndarray();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(owned_array[[i, j]], ref_array[[i, j]]);
            }
        }
    }
}
