pub use anyhow::{bail, Result};
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

/// Calculate a Bayesian posterior over a fixed support set.
///
/// This helper keeps the support points unchanged and updates only their weights:
/// `posterior_i ∝ prior_i × ∏_subjects likelihood(subject | theta_i)`.
///
/// The result is normalized to sum to 1 and the returned objective is the
/// log-evidence of the weighted support set.
#[allow(dead_code)]
pub fn weighted_support_posterior(psi: &Psi, prior_weights: &Weights) -> Result<(Weights, f64)> {
    if psi.matrix().ncols() != prior_weights.weights().nrows() {
        bail!(
            "Number of support points in psi ({}) and prior weights ({}) do not match.",
            psi.matrix().ncols(),
            prior_weights.weights().nrows()
        );
    }

    if psi.matrix().ncols() == 0 {
        bail!("Cannot compute a posterior for an empty support set.");
    }

    let psi_matrix = psi.matrix();
    let mut log_posterior = Vec::with_capacity(psi_matrix.ncols());

    for support_index in 0..psi_matrix.ncols() {
        let prior = prior_weights[support_index];

        if !prior.is_finite() || prior < 0.0 {
            bail!(
                "Prior weights must be finite and non-negative; found {} at index {}.",
                prior,
                support_index
            );
        }

        if prior == 0.0 {
            log_posterior.push(f64::NEG_INFINITY);
            continue;
        }

        let mut log_weight = prior.ln();
        for subject_index in 0..psi_matrix.nrows() {
            let likelihood = *psi_matrix.get(subject_index, support_index);

            if !likelihood.is_finite() || likelihood < 0.0 {
                bail!(
                    "Psi must contain finite non-negative likelihoods; found {} at row {}, column {}.",
                    likelihood,
                    subject_index,
                    support_index
                );
            }

            if likelihood == 0.0 {
                log_weight = f64::NEG_INFINITY;
                break;
            }

            log_weight += likelihood.ln();
        }

        log_posterior.push(log_weight);
    }

    let max_log_weight = log_posterior
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if !max_log_weight.is_finite() {
        bail!(
            "Posterior normalization failed because every support point has zero prior-weighted likelihood."
        );
    }

    let unnormalized: Vec<f64> = log_posterior
        .iter()
        .map(|log_weight| {
            if log_weight.is_finite() {
                (*log_weight - max_log_weight).exp()
            } else {
                0.0
            }
        })
        .collect();

    let normalizer: f64 = unnormalized.iter().sum();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        bail!("Posterior normalization failed because the evidence is not positive.");
    }

    Ok((
        Weights::from_vec(
            unnormalized
                .iter()
                .map(|weight| weight / normalizer)
                .collect(),
        ),
        max_log_weight + normalizer.ln(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{weighted_support_posterior, Psi};
    use crate::estimation::nonparametric::Weights;
    use approx::assert_relative_eq;
    use faer::Mat;

    #[test]
    fn weighted_support_posterior_respects_prior_weights() {
        let psi = Psi::from(Mat::from_fn(1, 3, |_row, _col| 0.5));
        let prior = Weights::from_vec(vec![0.9, 0.09, 0.01]);

        let (posterior, log_evidence) = weighted_support_posterior(&psi, &prior).unwrap();

        assert_relative_eq!(posterior[0], 0.9, epsilon = 1e-12);
        assert_relative_eq!(posterior[1], 0.09, epsilon = 1e-12);
        assert_relative_eq!(posterior[2], 0.01, epsilon = 1e-12);
        assert_relative_eq!(log_evidence, (0.5_f64).ln(), epsilon = 1e-12);
    }
}
