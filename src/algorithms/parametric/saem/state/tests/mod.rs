use super::diagnostics::{begin_retained_transition_accounting, worst_valid_max_rhat};
use super::*;
use crate::algorithms::parametric::{NumericalFailurePhase, ParametricRunner};
use crate::estimation::parametric::information::derive_population_uncertainty;
use crate::estimation::parametric::transforms::{phi_to_psi, psi_to_phi};
use crate::estimation::parametric::ParametricPrior;
use crate::estimation::{EstimationProblem, Iov, Omega, ParametricErrorModel};
use crate::model::Parameter;
use crate::results::{
    FitResult, PopulationUncertaintyDiagnostics, PopulationUncertaintyRegularization,
    PopulationUncertaintyStatus,
};
use pharmsol::prelude::*;
use pharmsol::SubjectBuilderExt;

#[test]
fn finite_improvement_is_eligible_without_a_convergence_flag() {
    assert!(non_iiv_candidate_improves(10.0, 9.0));
    assert!(!non_iiv_candidate_improves(10.0, 10.0));
    assert!(!non_iiv_candidate_improves(10.0, f64::NAN));
}

#[test]
fn censored_information_failure_has_explicit_unsupported_status() {
    let reason = "analytic information is unsupported for censored observations".to_string();
    assert_eq!(
        information_failure_status(reason.clone()),
        InformationStatus::Unsupported(reason)
    );
}

fn one_compartment_metadata() -> pharmsol::equation::ModelMetadata {
    equation::metadata::new("one_compartment_saem")
        .parameters(["ke", "v"])
        .states(["central"])
        .outputs(["0"])
        .route(equation::Route::bolus("0").to_state("central"))
}

fn one_compartment() -> pharmsol::equation::ODE {
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

fn sparse_second_output_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    let equation = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v);
            y[0] = x[0];
            y[1] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(2)
    .with_metadata(
        equation::metadata::new("sparse_second_output")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["unmeasured", "measured"])
            .route(equation::Route::bolus("dose").to_state("central")),
    )
    .unwrap();
    let data = Data::new(vec![Subject::builder("sparse")
        .bolus(0.0, 100.0, "dose")
        .observation(1.0, 8.0, "measured")
        .observation(2.0, 6.0, "measured")
        .build()]);

    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.2)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("measured", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn mixed_residual_output_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    let equation = equation::ODE::new(
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
            y[1] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(2)
    .with_metadata(
        equation::metadata::new("mixed_residual_outputs")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["fixed", "mixed"])
            .route(equation::Route::bolus("dose").to_state("central")),
    )
    .expect("mixed residual equation metadata should validate");
    let data = Data::new(vec![Subject::builder("mixed")
        .bolus(0.0, 100.0, "dose")
        .observation(1.0, 8.5, "fixed")
        .observation(2.0, 6.5, "fixed")
        .observation(1.0, 8.0, "mixed")
        .observation(2.0, 6.0, "mixed")
        .build()]);

    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.2)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "fixed",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.5)).fixed(),
        )
        .error_model(
            "mixed",
            ParametricErrorModel::new(ResidualErrorModel::combined(0.0, 0.1))
                .fixed_combined_additive(),
        )
        .build()
        .expect("mixed residual output problem should validate")
}

fn data() -> Data {
    Data::new(vec![
        Subject::builder("s1")
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 12.0, "0")
            .observation(4.0, 4.0, "0")
            .build(),
        Subject::builder("s2")
            .bolus(0.0, 80.0, "0")
            .observation(0.5, 9.0, "0")
            .observation(3.0, 2.5, "0")
            .build(),
    ])
}

fn covariate_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    let subjects = [-1.0, 0.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(index, wt)| {
            Subject::builder(format!("cov{index}"))
                .covariate("wt", 0.0, wt)
                .covariate("sex", 0.0, if index == 2 { 1.0 } else { 0.0 })
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 8.0 + index as f64, "0")
                .build()
        })
        .collect();
    EstimationProblem::parametric(one_compartment(), Data::new(subjects))
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .covariate_effect(
            crate::estimation::parametric::CovariateEffect::continuous("ke", "wt", 0.0)
                .with_initial(0.0),
        )
        .covariate_effect(
            crate::estimation::parametric::CovariateEffect::categorical("v", "sex", 0.0, 1.0)
                .with_initial(0.0),
        )
        .error_model(
            "0",
            ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
        )
        .build()
        .unwrap()
}

fn fixed_covariate_iiv_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    let subjects = [-1.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(index, wt)| {
            Subject::builder(format!("fixed-cov-iiv-{index}"))
                .covariate("wt", 0.0, wt)
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 5.0 + index as f64, "0")
                .build()
        })
        .collect();
    EstimationProblem::parametric(one_compartment(), Data::new(subjects))
        .parameter(Parameter::log("ke").with_initial(0.2).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::diagonal([("ke", 1.0)]))
        .covariate_effect(
            crate::estimation::parametric::CovariateEffect::continuous("ke", "wt", 0.0)
                .with_initial(0.0)
                .fixed(),
        )
        .error_model(
            "0",
            ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
        )
        .build()
        .unwrap()
}

fn fixed_covariate_without_iiv_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    let subjects = [0.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(index, wt)| {
            Subject::builder(format!("fixed-cov-{index}"))
                .covariate("wt", 0.0, wt)
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 5.0 + index as f64, "0")
                .build()
        })
        .collect();
    EstimationProblem::parametric(one_compartment(), Data::new(subjects))
        .parameter(
            Parameter::log("ke")
                .with_initial(0.2)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .covariate_effect(
            crate::estimation::parametric::CovariateEffect::continuous("ke", "wt", 0.0)
                .with_initial(0.2)
                .fixed(),
        )
        .error_model(
            "0",
            ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
        )
        .build()
        .unwrap()
}

fn problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .error_model(
            "0",
            ParametricErrorModel::new(ResidualErrorModel::combined(0.5, 0.1)).fixed(),
        )
        .build()
        .unwrap()
}

fn constant_error_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn partial_iiv_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn iov_data() -> Data {
    Data::new(vec![Subject::builder("s1")
        .bolus(0.0, 100.0, "0")
        .observation(1.0, 12.0, "0")
        .reset()
        .bolus(0.0, 100.0, "0")
        .observation(1.0, 10.0, "0")
        .build()])
}

fn iov_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), iov_data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .iov(Iov::diagonal([("ke", 0.1)]))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn markov_iov_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), iov_data())
        .parameter(Parameter::log("ke").with_initial(0.2).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.1))
        .iov(Iov::new().fixed_variance("ke", 0.1))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn uneven_iov_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    let data = Data::new(vec![
        Subject::builder("one")
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 12.0, "0")
            .build(),
        Subject::builder("two")
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 12.0, "0")
            .reset()
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 10.0, "0")
            .build(),
        Subject::builder("three")
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 12.0, "0")
            .reset()
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 10.0, "0")
            .reset()
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 11.0, "0")
            .build(),
    ]);
    EstimationProblem::parametric(one_compartment(), data)
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .iov(Iov::diagonal([("ke", 0.1)]))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn configured_iov_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), iov_data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .iov(
            Iov::diagonal([("ke", 0.10)])
                .fixed_variance("v", 0.20)
                .fixed_covariance("ke", "v", 0.05),
        )
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn ordered_metadata_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), iov_data())
        .parameter(Parameter::real("ke").with_initial(0.2))
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .iov(Iov::diagonal([("v", 0.20)]))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn configured_omega_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .omega(Omega::diagonal([("ke", 0.25)]).fixed_variance("v", 0.5))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn correlated_omega_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2))
        .parameter(Parameter::log("v").with_initial(10.0))
        .omega(Omega::diagonal([("ke", 0.25), ("v", 0.25)]).covariance("ke", "v", 0.20))
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn fixed_population_iiv_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(Parameter::log("ke").with_initial(0.2).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn fixed_no_iiv_problem() -> EstimationProblem<pharmsol::equation::ODE, Parametric> {
    EstimationProblem::parametric(one_compartment(), data())
        .parameter(
            Parameter::log("ke")
                .with_initial(0.2)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(10.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("0", ResidualErrorModel::constant(1.0))
        .build()
        .unwrap()
}

fn assert_prediction_points_equal(
    actual: &pharmsol::simulator::prediction::SubjectPredictions,
    expected: &pharmsol::simulator::prediction::SubjectPredictions,
) {
    assert_eq!(actual.predictions().len(), expected.predictions().len());
    assert_eq!(actual.occasions(), expected.occasions());
    for (actual, expected) in actual.predictions().iter().zip(expected.predictions()) {
        assert_eq!(actual.time(), expected.time());
        assert_eq!(actual.observation(), expected.observation());
        assert_eq!(actual.prediction(), expected.prediction());
        assert_eq!(actual.output(), expected.output());
        assert_eq!(actual.errorpoly(), expected.errorpoly());
        assert_eq!(actual.censoring(), expected.censoring());
    }
}

mod controller;
mod diagnostics;
mod estimation;
mod results;
mod schedule;
mod state_and_iov;
