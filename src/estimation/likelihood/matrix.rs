use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis, ShapeBuilder};
use pharmsol::{Data, Equation};

use crate::AssayErrorModels;
use rayon::prelude::*;

use super::observation::{assay_error_model_log_likelihoods, AssayLikelihoodError};

/// Compute a nonparametric log-likelihood matrix with shape
/// `(n_subjects, n_support_points)`.
pub(crate) fn nonparametric_log_likelihood_matrix(
    equation: &impl Equation,
    subjects: &Data,
    support_points: &Array2<f64>,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Array2<f64>> {
    let n_support_points = support_points.nrows();
    let subject_refs = subjects.subjects();
    let support_point_rows = support_points
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();

    if progress {
        println!(
            "Computing log-likelihood matrix: {} subjects × {} support points...",
            subject_refs.len(),
            n_support_points
        );
    }

    let mut log_psi: Array2<f64> = Array2::default((subjects.len(), n_support_points).f());

    let result: Result<()> = log_psi
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            let subject = subject_refs[i];

            for (element, support_point) in row.iter_mut().zip(support_point_rows.iter()) {
                let predictions = equation
                    .estimate_predictions_dense(subject, support_point.as_slice())
                    .map_err(|err| anyhow!(err))?;
                *element = match assay_error_model_log_likelihoods(&predictions, error_models) {
                    Ok(score) => score,
                    Err(AssayLikelihoodError::Impossible) => f64::NEG_INFINITY,
                    Err(error) => return Err(error.into()),
                };
            }

            Ok(())
        });

    result?;

    if progress {
        println!("Log-likelihood matrix complete.");
    }

    Ok(log_psi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssayErrorModel, ErrorPoly};
    use pharmsol::prelude::*;
    use pharmsol::{Censor, SubjectBuilderExt};

    fn one_compartment_metadata() -> pharmsol::equation::ModelMetadata {
        equation::metadata::new("one_compartment_likelihood_parity")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("central"))
    }

    fn one_compartment() -> pharmsol::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(one_compartment_metadata())
        .unwrap()
    }

    fn direct_output() -> pharmsol::ODE {
        equation::ODE::new(
            |_x, _p, _t, dx, _b, _rateiv, _cov| dx[0] = 0.0,
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |_x, p, _t, _cov, y| y[0] = p[0],
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            equation::metadata::new("direct_output_likelihood_matrix")
                .parameters(["value"])
                .states(["state"])
                .outputs(["0"])
                .route(equation::Route::bolus("dose").to_state("state")),
        )
        .unwrap()
    }

    fn parity_data() -> Data {
        let s1 = Subject::builder("s1")
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 12.0, "0")
            .missing_observation(2.0, "0")
            .censored_observation(4.0, 1.0, "0", Censor::BLOQ)
            .build();
        let s2 = Subject::builder("s2")
            .bolus(0.0, 80.0, "0")
            .observation(0.5, 9.0, "0")
            .censored_observation(3.0, 15.0, "0", Censor::ALOQ)
            .build();
        Data::new(vec![s1, s2])
    }

    fn error_models() -> AssayErrorModels {
        AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 0.0),
            )
            .unwrap()
    }

    #[test]
    fn matrix_scores_all_subject_support_point_pairs() {
        let equation = one_compartment();
        let data = parity_data();
        let support_points = ndarray::array![[0.15, 8.0], [0.30, 12.0], [0.55, 20.0]];
        let error_models = error_models();

        let scores = nonparametric_log_likelihood_matrix(
            &equation,
            &data,
            &support_points,
            &error_models,
            false,
        )
        .expect("pmcore matrix");

        assert_eq!(scores.dim(), (2, 3));
        assert!(scores.iter().all(|score| score.is_finite()));
    }

    #[test]
    fn matrix_preserves_impossible_cells_as_negative_infinity() {
        let equation = direct_output();
        let data = Data::new(vec![Subject::builder("s1")
            .observation(1.0, 1e155, "0")
            .build()]);
        let support_points = ndarray::array![[1e155], [0.0]];
        let error_models = AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(1.0, 0.0, 0.0, 0.0), 0.0),
            )
            .unwrap();

        let scores = nonparametric_log_likelihood_matrix(
            &equation,
            &data,
            &support_points,
            &error_models,
            false,
        )
        .expect("an impossible support point must not abort matrix construction");

        assert!(scores[(0, 0)].is_finite());
        assert_eq!(scores[(0, 1)], f64::NEG_INFINITY);
    }
}
