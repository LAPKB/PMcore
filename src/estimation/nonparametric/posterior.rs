pub use anyhow::{Result, bail};
use faer::Mat;
use serde::{Deserialize, Serialize};

use crate::estimation::nonparametric::{psi::Psi, weights::Weights};

#[derive(Debug, Clone)]
pub struct Posterior {
    mat: Mat<f64>,
}

impl Posterior {
    fn new(mat: Mat<f64>) -> Self {
        Posterior { mat }
    }

    pub fn calculate(psi: &Psi, w: &Weights) -> Result<Self> {
        if psi.matrix().ncols() != w.weights().nrows() {
            bail!(
                "Number of rows in psi ({}) and number of weights ({}) do not match.",
                psi.matrix().nrows(),
                w.weights().nrows()
            );
        }

        let psi_matrix = psi.matrix();
        let py = psi_matrix * w.weights();

        let posterior = Mat::from_fn(psi_matrix.nrows(), psi_matrix.ncols(), |i, j| {
            psi_matrix.get(i, j) * w.weights().get(j) / py.get(i)
        });

        Ok(posterior.into())
    }

    pub fn matrix(&self) -> &Mat<f64> {
        &self.mat
    }

    pub fn to_csv<W: std::io::Write>(&self, writer: W) -> Result<()> {
        let mut csv_writer = csv::Writer::from_writer(writer);

        for i in 0..self.mat.nrows() {
            let row: Vec<f64> = (0..self.mat.ncols()).map(|j| *self.mat.get(i, j)).collect();
            csv_writer.serialize(row)?;
        }

        csv_writer.flush()?;
        Ok(())
    }

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

        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                bail!("Row {} has {} columns, expected {}", i, row.len(), ncols);
            }
        }

        let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);

        Ok(Posterior::new(mat))
    }
}

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

                let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);

                Ok(Posterior::new(mat))
            }
        }

        deserializer.deserialize_seq(PosteriorVisitor)
    }
}

pub fn posterior(psi: &Psi, w: &Weights) -> Result<Posterior> {
    Posterior::calculate(psi, w)
}