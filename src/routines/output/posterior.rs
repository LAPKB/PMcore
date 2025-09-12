pub use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

use crate::structs::{psi::Psi, weights::Weights};

/// Posterior probabilities for each support points
#[derive(Debug, Clone)]
pub struct Posterior {
    mat: Mat<f64>,
}

impl Posterior {
    /// Create a new Posterior from a matrix
    fn new(mat: Mat<f64>) -> Self {
        Posterior { mat }
    }

    /// Calculate the posterior probabilities for each support point given the weights
    ///
    /// The shape is the same as [Psi], and thus subjects are the rows and support points are the columns.
    /// /// # Errors
    /// Returns an error if the number of rows in `psi` does not match the number of weights in `w`.
    /// # Arguments
    /// * `psi` - The Psi object containing the matrix of support points.
    /// * `w` - The weights for each support point.
    /// # Returns
    /// A Result containing the Posterior probabilities if successful, or an error if the
    /// dimensions do not match.
    pub fn calculate(psi: &Psi, w: &Col<f64>) -> Result<Self> {
        if psi.matrix().ncols() != w.nrows() {
            bail!(
                "Number of rows in psi ({}) and number of weights ({}) do not match.",
                psi.matrix().nrows(),
                w.nrows()
            );
        }

        let psi_matrix = psi.matrix();
        let py = psi_matrix * w;

        let posterior = Mat::from_fn(psi_matrix.nrows(), psi_matrix.ncols(), |i, j| {
            psi_matrix.get(i, j) * w.get(j) / py.get(i)
        });

        Ok(posterior.into())
    }

    /// Get a reference to the underlying matrix
    pub fn matrix(&self) -> &Mat<f64> {
        &self.mat
    }

    /// Write the posterior probabilities to a CSV file
    /// Each row represents a subject, each column represents a support point
    pub fn to_csv<W: std::io::Write>(&self, writer: W) -> Result<()> {
        let mut csv_writer = csv::Writer::from_writer(writer);

        // Write each row
        for i in 0..self.mat.nrows() {
            let row: Vec<f64> = (0..self.mat.ncols()).map(|j| *self.mat.get(i, j)).collect();
            csv_writer.serialize(row)?;
        }

        csv_writer.flush()?;
        Ok(())
    }

    /// Read posterior probabilities from a CSV file
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

        Ok(Posterior::new(mat))
    }
}

/// Convert a matrix to a [Posterior]
impl From<Mat<f64>> for Posterior {
    fn from(mat: Mat<f64>) -> Self {
        Posterior::new(mat)
    }
}

impl Serialize for Posterior {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(self.mat.nrows()))?;

        // Serialize each row as a vector
        for i in 0..self.mat.nrows() {
            let row: Vec<f64> = (0..self.mat.ncols()).map(|j| *self.mat.get(i, j)).collect();
            seq.serialize_element(&row)?;
        }

        seq.end()
    }
}

impl<'de> Deserialize<'de> for Posterior {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{SeqAccess, Visitor};
        use std::fmt;

        struct PosteriorVisitor;

        impl<'de> Visitor<'de> for PosteriorVisitor {
            type Value = Posterior;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of rows (vectors of f64)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
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

                Ok(Posterior::new(mat))
            }
        }

        deserializer.deserialize_seq(PosteriorVisitor)
    }
}

/// Calculates the posterior probabilities for each support point given the weights
///
/// The shape is the same as [Psi], and thus subjects are the rows and support points are the columns.
pub fn posterior(psi: &Psi, w: &Weights) -> Result<Posterior> {
    if psi.matrix().ncols() != w.len() {
        bail!(
            "Number of rows in psi ({}) and number of weights ({}) do not match.",
            psi.matrix().nrows(),
            w.len()
        );
    }

    let psi_matrix = psi.matrix();
    let py = psi_matrix * w.weights();

    let posterior = Mat::from_fn(psi_matrix.nrows(), psi_matrix.ncols(), |i, j| {
        psi_matrix.get(i, j) * w.weights().get(j) / py.get(i)
    });

    Ok(posterior.into())
}
