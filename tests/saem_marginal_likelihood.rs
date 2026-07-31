use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static N2_EQUATION_CALLS: AtomicUsize = AtomicUsize::new(0);

use pharmsol::prelude::*;
use pharmsol::Cache;
use pmcore::prelude::*;

fn latent_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n2_latent_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .observation(3.0, 3.1, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.2, "cp")
            .observation(3.0, 3.6, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.25).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.09))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.35)).fixed(),
        )
        .build()
        .expect("latent N2 fixture")
}

fn joint_iiv_iov_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n2_joint_iiv_iov_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp, amount],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
            y[amount] = x[central];
        },
    };
    let one = Subject::builder("one")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.9, "cp")
        .observation(1.0, 98.0, "amount")
        .build();
    let two = Subject::builder("two")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 5.1, "cp")
        .observation(1.0, 101.0, "amount")
        .reset()
        .infusion(12.0, 100.0, "iv", 0.5)
        .observation(13.0, 4.7, "cp")
        .observation(13.0, 96.0, "amount")
        .build();
    let three = Subject::builder("three")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 5.0, "cp")
        .observation(1.0, 100.0, "amount")
        .reset()
        .infusion(12.0, 100.0, "iv", 0.5)
        .observation(13.0, 5.2, "cp")
        .observation(13.0, 103.0, "amount")
        .reset()
        .infusion(24.0, 100.0, "iv", 0.5)
        .observation(25.0, 4.8, "cp")
        .observation(25.0, 97.0, "amount")
        .build();
    EstimationProblem::parametric(equation, Data::new(vec![one, two, three]))
        .parameter(Parameter::log("ke").with_initial(0.25).fixed())
        .parameter(Parameter::log("v").with_initial(20.0).fixed())
        .omega(
            Omega::new()
                .fixed_variance("ke", 0.09)
                .fixed_variance("v", 0.16)
                .fixed_covariance("ke", "v", 0.03),
        )
        .iov(
            Iov::new()
                .fixed_variance("ke", 0.04)
                .fixed_variance("v", 0.09)
                .fixed_covariance("ke", "v", 0.01),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.3)).fixed(),
        )
        .error_model(
            "amount",
            ParametricErrorModel::new(ResidualErrorModel::constant(4.0)).fixed(),
        )
        .build()
        .expect("joint IIV+IOV N2 fixture")
}

fn no_latent_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n2_no_latent_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.1, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new())
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.35)).fixed(),
        )
        .build()
        .expect("no-latent N2 fixture")
}

fn instrumented_no_latent_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric>
{
    let equation = analytical! {
        name: "n2_instrumented_no_latent_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            N2_EQUATION_CALLS.fetch_add(1, Ordering::SeqCst);
            y[cp] = x[central] / v;
        },
    };
    let equation = equation.disable_cache();
    let data = Data::new(vec![
        Subject::builder("counter-1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .build(),
        Subject::builder("counter-2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.1, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new())
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.35)).fixed(),
        )
        .build()
        .expect("instrumented no-latent N2 fixture")
}

fn fit_config() -> SaemConfig {
    SaemConfig::new()
        .seed(7001)
        .n_chains(2)
        .mcmc_iterations(2)
        .burn_in(0)
        .k1_iterations(2)
        .k2_iterations(1)
        .compute_map(true)
}

fn n2(seed: u64) -> MarginalLikelihoodConfig {
    MarginalLikelihoodConfig::new(1024, seed, 5, 1.5)
}

fn write_record(path: &std::path::Path, record: &ParametricResultRecord) {
    fs::write(path, serde_json::to_vec_pretty(record).unwrap()).unwrap();
}

fn clear_subject_n2_numerics(subject: &mut SubjectMarginalLikelihoodDiagnostics) {
    subject.log_marginal_likelihood = None;
    subject.n2ll = None;
    subject.effective_sample_size = None;
    subject.effective_sample_fraction = None;
    subject.var_log = None;
    subject.n2ll_mcse = None;
    subject.zero_weight_count = 0;
}

fn synchronize_unavailable_n2_tables(record: &mut ParametricResultRecord) {
    let diagnostics = record.marginal_likelihood.as_ref().unwrap();
    let failures = diagnostics
        .subjects
        .iter()
        .filter_map(|subject| {
            subject
                .failure
                .clone()
                .map(|reason| MarginalLikelihoodSubjectFailure {
                    subject_id: subject.subject_id.clone(),
                    reason,
                })
        })
        .collect::<Vec<_>>();
    record.marginal_likelihood.as_mut().unwrap().status = MarginalLikelihoodStatus::Unavailable {
        failures: failures.clone(),
    };
    let diagnostics = record.marginal_likelihood.as_ref().unwrap();
    let total = record
        .tables
        .marginal_likelihood
        .iter_mut()
        .find(|row| row.scope == "total")
        .unwrap();
    total.status = "unavailable".to_string();
    total.log_marginal_likelihood = None;
    total.n2ll = None;
    total.n2ll_mcse = None;
    total.zero_weight_count = diagnostics
        .subjects
        .iter()
        .map(|subject| subject.zero_weight_count)
        .sum();
    total.failure = Some(serde_json::to_string(&failures).unwrap());
    for subject in &diagnostics.subjects {
        let row = record
            .tables
            .marginal_likelihood
            .iter_mut()
            .find(|row| row.subject.as_deref() == Some(subject.subject_id.as_str()))
            .unwrap();
        row.status = if subject.failure.is_some() {
            "unavailable".to_string()
        } else if subject.mode_converged == Some(false) {
            "available_with_nonconverged_mode".to_string()
        } else {
            "available".to_string()
        };
        row.mode = serde_json::to_string(&subject.mode).unwrap();
        row.mode_converged = subject.mode_converged;
        row.log_marginal_likelihood = subject.log_marginal_likelihood;
        row.n2ll = subject.n2ll;
        row.n2ll_mcse = subject.n2ll_mcse;
        row.effective_sample_size = subject.effective_sample_size;
        row.effective_sample_fraction = subject.effective_sample_fraction;
        row.zero_weight_count = subject.zero_weight_count;
        row.failure = subject
            .failure
            .as_ref()
            .map(|reason| serde_json::to_string(reason).unwrap());
    }
    for row in record
        .tables
        .statistics
        .iter_mut()
        .filter(|row| row.kind.starts_with("marginal_likelihood"))
    {
        if row.kind == "marginal_likelihood_status" || row.kind == "marginal_likelihood" {
            row.status = Some("unavailable".to_string());
        }
        if row.kind == "marginal_likelihood" {
            row.value = None;
        } else if row.kind == "marginal_likelihood_subject_status" {
            let subject = diagnostics
                .subjects
                .iter()
                .find(|subject| subject.subject_id == row.name)
                .unwrap();
            row.value = subject.n2ll;
            row.status = Some(if subject.failure.is_some() {
                "unavailable".to_string()
            } else {
                "available".to_string()
            });
        }
    }

    let reason = InformationCriteriaUnavailableReason::SourceMarginalLikelihoodUnavailable;
    record.information_criteria.status = InformationCriteriaStatus::Unavailable {
        reason: reason.clone(),
    };
    record.information_criteria.source_marginal_n2ll = None;
    record.information_criteria.source_marginal_n2ll_mcse = None;
    record.information_criteria.aic = None;
    record.information_criteria.bic = None;
    record.information_criteria.aic_mcse = None;
    record.information_criteria.bic_mcse = None;
    let criteria = record.tables.information_criteria.first_mut().unwrap();
    criteria.status = "unavailable".to_string();
    criteria.source_marginal_n2ll = None;
    criteria.source_marginal_n2ll_mcse = None;
    criteria.aic = None;
    criteria.bic = None;
    criteria.aic_mcse = None;
    criteria.bic_mcse = None;
    criteria.failure_reason = Some(serde_json::to_string(&reason).unwrap());
    for row in record
        .tables
        .statistics
        .iter_mut()
        .filter(|row| row.kind.starts_with("information_criteria"))
    {
        row.status = Some("unavailable".to_string());
        if row.kind == "information_criteria" {
            row.value = None;
        }
    }
}

fn coordinate_retained_n2_config(
    record: &mut ParametricResultRecord,
    config: MarginalLikelihoodConfig,
) {
    record.config.marginal_likelihood = Some(config);
    let diagnostics = record.marginal_likelihood.as_mut().unwrap();
    diagnostics.config = config;
    for subject in &mut diagnostics.subjects {
        if subject.method == MarginalLikelihoodMethod::StudentTImportanceSampling {
            subject.samples = config.samples_per_subject;
            if config.samples_per_subject == 1 && subject.failure.is_none() {
                subject.effective_sample_size = Some(1.0);
                subject.effective_sample_fraction = Some(1.0);
            }
        }
    }
    for row in &mut record.tables.marginal_likelihood {
        row.samples_per_subject = if row.scope == "total" {
            config.samples_per_subject
        } else {
            diagnostics
                .subjects
                .iter()
                .find(|subject| subject.subject_id == row.subject.clone().unwrap())
                .unwrap()
                .samples
        };
        row.degrees_of_freedom = config.degrees_of_freedom;
        row.covariance_scale_multiplier = config.covariance_scale_multiplier;
        if let Some(subject_id) = row.subject.as_deref() {
            let subject = diagnostics
                .subjects
                .iter()
                .find(|subject| subject.subject_id == subject_id)
                .unwrap();
            row.effective_sample_size = subject.effective_sample_size;
            row.effective_sample_fraction = subject.effective_sample_fraction;
        }
    }
}

#[test]
fn no_latent_n2_is_exact_without_map_or_fabricated_ess() {
    let result = no_latent_problem()
        .fit_with(
            fit_config()
                .compute_map(false)
                .marginal_likelihood(n2(8001)),
        )
        .expect("exact N2 fit");
    let diagnostics = result
        .marginal_likelihood_diagnostics()
        .expect("N2 diagnostics");
    assert!(matches!(
        diagnostics.status,
        MarginalLikelihoodStatus::Available
    ));
    assert!((result.marginal_n2ll().unwrap() - result.conditional_n2ll()).abs() <= 1e-10);
    assert_eq!(result.marginal_n2ll_mcse(), Some(0.0));
    for subject in &diagnostics.subjects {
        assert_eq!(subject.method, MarginalLikelihoodMethod::ExactNoLatent);
        assert_eq!(subject.samples, 0);
        assert_eq!(subject.n2ll_mcse, Some(0.0));
        assert_eq!(subject.effective_sample_size, None);
        assert_eq!(subject.effective_sample_fraction, None);
    }
}

#[test]
fn enabled_exact_n2_performs_only_the_explicit_post_fit_scoring_calls() {
    N2_EQUATION_CALLS.store(0, Ordering::SeqCst);
    let disabled = instrumented_no_latent_problem()
        .fit_with(fit_config().compute_map(false))
        .expect("instrumented disabled fit");
    let disabled_calls = N2_EQUATION_CALLS.load(Ordering::SeqCst);

    N2_EQUATION_CALLS.store(0, Ordering::SeqCst);
    let enabled = instrumented_no_latent_problem()
        .fit_with(
            fit_config()
                .compute_map(false)
                .marginal_likelihood(n2(8051)),
        )
        .expect("instrumented enabled fit");
    let enabled_calls = N2_EQUATION_CALLS.load(Ordering::SeqCst);

    assert_eq!(disabled.cycle_diagnostics(), enabled.cycle_diagnostics());
    assert_eq!(
        disabled.population_parameters(),
        enabled.population_parameters()
    );
    assert_eq!(disabled.conditional_n2ll(), enabled.conditional_n2ll());
    assert_eq!(
        enabled_calls - disabled_calls,
        enabled.data().subjects().len(),
        "exact no-latent N2 must add one post-fit scoring call per subject"
    );
}

#[test]
fn n2_stream_is_reproducible_and_does_not_change_canonical_fit() {
    let disabled = latent_problem()
        .fit_with(fit_config())
        .expect("disabled fit");
    let first = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8101)))
        .expect("first N2 fit");
    let repeated = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8101)))
        .expect("repeated N2 fit");
    let changed = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8102)))
        .expect("changed-seed N2 fit");

    assert_eq!(disabled.cycle_diagnostics(), first.cycle_diagnostics());
    assert_eq!(
        disabled.population_parameters(),
        first.population_parameters()
    );
    assert_eq!(disabled.conditional_n2ll(), first.conditional_n2ll());
    assert_eq!(
        first.marginal_likelihood_diagnostics(),
        repeated.marginal_likelihood_diagnostics()
    );
    assert_ne!(
        first.marginal_likelihood_diagnostics(),
        changed.marginal_likelihood_diagnostics()
    );
    assert_eq!(first.cycle_diagnostics(), changed.cycle_diagnostics());
    assert_eq!(
        first.population_parameters(),
        changed.population_parameters()
    );
    assert_eq!(first.conditional_n2ll(), changed.conditional_n2ll());
    assert!(first.marginal_n2ll().is_some());
    assert!(first.marginal_n2ll_mcse().is_some());
    assert_eq!(first.summary().marginal_n2ll, first.marginal_n2ll());
}

#[test]
fn fourfold_sample_budget_reduces_median_reported_mcse_at_frozen_rate() {
    let mut low = Vec::new();
    let mut high = Vec::new();
    for seed in 8501..8511 {
        let low_result = latent_problem()
            .fit_with(
                fit_config().marginal_likelihood(MarginalLikelihoodConfig::new(4096, seed, 5, 1.5)),
            )
            .expect("low-budget N2 fit");
        let high_result = latent_problem()
            .fit_with(
                fit_config()
                    .marginal_likelihood(MarginalLikelihoodConfig::new(16384, seed, 5, 1.5)),
            )
            .expect("high-budget N2 fit");
        low.push(low_result.marginal_n2ll_mcse().expect("low MCSE"));
        high.push(high_result.marginal_n2ll_mcse().expect("high MCSE"));
    }
    low.sort_by(f64::total_cmp);
    high.sort_by(f64::total_cmp);
    let low_median = (low[4] + low[5]) / 2.0;
    let high_median = (high[4] + high[5]) / 2.0;
    eprintln!(
        "N2 MCSE medians: K4096={low_median:.17}, K16384={high_median:.17}, ratio={:.17}",
        high_median / low_median
    );
    assert!(
        high_median <= 0.65 * low_median,
        "high median {high_median} exceeds frozen 0.65 ratio of low median {low_median}"
    );
}

#[test]
fn joint_iiv_iov_uses_correlated_blocks_and_actual_uneven_occasion_order() {
    let result = joint_iiv_iov_problem()
        .fit_with(
            fit_config().marginal_likelihood(MarginalLikelihoodConfig::new(512, 8151, 5, 1.5)),
        )
        .expect("joint IIV+IOV N2 fit");
    let diagnostics = result
        .marginal_likelihood_diagnostics()
        .expect("joint diagnostics");
    assert!(result.marginal_n2ll().is_some());
    assert!(result.marginal_n2ll_mcse().is_some());
    assert_eq!(diagnostics.subjects.len(), 3);
    for (subject, data_subject) in diagnostics.subjects.iter().zip(result.data().subjects()) {
        let expected = data_subject
            .occasions()
            .iter()
            .map(|occasion| occasion.index())
            .collect::<Vec<_>>();
        assert_eq!(subject.occasion_indices, expected);
        assert_eq!(subject.dimension, 2 + 2 * expected.len());
        assert_eq!(
            subject.proposal_scale_source,
            ProposalScaleSource::FinalRawOmegaBlocks
        );
        assert!(subject.effective_sample_size.is_some());
        assert!(subject.effective_sample_fraction.is_some());
    }
}

#[test]
fn latent_n2_without_map_fails_before_fit() {
    let error = latent_problem()
        .fit_with(
            fit_config()
                .compute_map(false)
                .marginal_likelihood(n2(8201)),
        )
        .expect_err("latent N2 without MAP must fail");
    assert!(format!("{error:#}").contains("requires compute_map=true"));
}

#[test]
fn schema_six_persists_complete_n2_and_warm_start_recomputes_only_on_request() {
    let parent = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8301)))
        .expect("parent N2 fit");
    let directory = std::env::temp_dir().join(format!(
        "pmcore-n2-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    parent
        .write_outputs(&directory, 0.0, 0.0)
        .expect("write N2 outputs");
    let record = ParametricResultRecord::read_json(directory.join("result.json"))
        .expect("read schema-six N2 result");
    assert_eq!(record.schema_version, 9);
    assert_eq!(
        record.marginal_likelihood.as_ref(),
        parent.marginal_likelihood_diagnostics()
    );
    assert_eq!(
        record.tables.marginal_likelihood.len(),
        parent.data().subjects().len() + 1
    );
    let csv = fs::read_to_string(directory.join("marginal_likelihood.csv"))
        .expect("read marginal likelihood CSV");
    assert_eq!(csv.lines().count(), parent.data().subjects().len() + 2);
    let csv_rows = csv::Reader::from_reader(csv.as_bytes())
        .deserialize::<MarginalLikelihoodRow>()
        .collect::<Result<Vec<_>, _>>()
        .expect("deserialize marginal-likelihood CSV rows");
    assert_eq!(csv_rows, record.tables.marginal_likelihood);
    let diagnostics = record.marginal_likelihood.as_ref().unwrap();
    assert_eq!(
        record.tables.marginal_likelihood[0].log_marginal_likelihood,
        diagnostics.log_marginal_likelihood
    );
    assert_eq!(record.tables.marginal_likelihood[0].n2ll, diagnostics.n2ll);
    assert_eq!(
        record.tables.marginal_likelihood[0].n2ll_mcse,
        diagnostics.n2ll_mcse
    );
    for (row, subject) in record.tables.marginal_likelihood[1..]
        .iter()
        .zip(&diagnostics.subjects)
    {
        assert_eq!(row.subject.as_deref(), Some(subject.subject_id.as_str()));
        assert_eq!(row.log_marginal_likelihood, subject.log_marginal_likelihood);
        assert_eq!(row.n2ll, subject.n2ll);
        assert_eq!(row.n2ll_mcse, subject.n2ll_mcse);
        assert_eq!(row.effective_sample_size, subject.effective_sample_size);
        assert_eq!(
            row.effective_sample_fraction,
            subject.effective_sample_fraction
        );
    }
    for (name, expected) in [
        (
            "log_marginal_likelihood",
            diagnostics.log_marginal_likelihood,
        ),
        ("marginal_n2ll", diagnostics.n2ll),
        ("marginal_n2ll_mcse", diagnostics.n2ll_mcse),
    ] {
        assert_eq!(
            record
                .tables
                .statistics
                .iter()
                .find(|row| row.kind == "marginal_likelihood" && row.name == name)
                .and_then(|row| row.value),
            expected
        );
    }

    let child_disabled = parent
        .fit_next(fit_config().seed(7002))
        .expect("disabled warm-start child");
    assert!(child_disabled.marginal_likelihood_diagnostics().is_none());
    let child_enabled = parent
        .fit_next(fit_config().seed(7002).marginal_likelihood(n2(8302)))
        .expect("enabled warm-start child");
    assert_eq!(child_enabled.config().marginal_likelihood, Some(n2(8302)));
    assert_eq!(parent.config().marginal_likelihood, Some(n2(8301)));
    fs::remove_dir_all(directory).expect("remove N2 output directory");
}

#[test]
fn schema_six_round_trips_global_posthoc_failure_without_fabricated_modes() {
    let result = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8351)))
        .expect("global failure persistence fixture");
    let path = std::env::temp_dir().join(format!(
        "pmcore-n2-global-posthoc-failure-{}.json",
        std::process::id()
    ));
    result.write_json(&path, 0.0, 0.0).expect("write fixture");
    let mut record = ParametricResultRecord::read_json(&path).expect("read fixture");
    let reason = MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(
        "global conditional mode calculation failed: optimizer fixture".to_string(),
    );
    let diagnostics = record.marginal_likelihood.as_mut().unwrap();
    diagnostics.log_marginal_likelihood = None;
    diagnostics.n2ll = None;
    diagnostics.n2ll_mcse = None;
    for subject in &mut diagnostics.subjects {
        subject.mode.clear();
        subject.mode_converged = None;
        clear_subject_n2_numerics(subject);
        subject.failure = Some(reason.clone());
    }
    synchronize_unavailable_n2_tables(&mut record);
    write_record(&path, &record);

    let round_trip =
        ParametricResultRecord::read_json(&path).expect("global posthoc failure should round trip");
    let diagnostics = round_trip.marginal_likelihood.unwrap();
    let expected_failures = diagnostics
        .subjects
        .iter()
        .map(|subject| MarginalLikelihoodSubjectFailure {
            subject_id: subject.subject_id.clone(),
            reason: reason.clone(),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        diagnostics.status,
        MarginalLikelihoodStatus::Unavailable {
            failures: expected_failures
        }
    );
    assert_eq!(diagnostics.log_marginal_likelihood, None);
    assert_eq!(diagnostics.n2ll, None);
    assert_eq!(diagnostics.n2ll_mcse, None);
    for (index, subject) in diagnostics.subjects.iter().enumerate() {
        assert_eq!(subject.dimension, 1);
        assert!(subject.occasion_indices.is_empty());
        assert_eq!(subject.samples, 1024);
        assert_eq!(
            subject.seed,
            Some(pmcore::estimation::parametric::marginal_likelihood_subject_seed(8351, index,))
        );
        assert!(subject.mode.is_empty());
        assert_eq!(subject.mode_converged, None);
        assert_eq!(subject.failure, Some(reason.clone()));
        assert!(subject.log_marginal_likelihood.is_none());
        assert!(subject.n2ll.is_none());
        assert!(subject.effective_sample_size.is_none());
        assert!(subject.effective_sample_fraction.is_none());
        assert!(subject.var_log.is_none());
        assert!(subject.n2ll_mcse.is_none());
    }
    fs::remove_file(path).expect("remove global failure fixture");
}

#[test]
fn schema_six_round_trips_missing_mode_for_non_first_subject() {
    let result = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8352)))
        .expect("missing mode persistence fixture");
    let path = std::env::temp_dir().join(format!(
        "pmcore-n2-missing-mode-{}.json",
        std::process::id()
    ));
    result.write_json(&path, 0.0, 0.0).expect("write fixture");
    let mut record = ParametricResultRecord::read_json(&path).expect("read fixture");
    let diagnostics = record.marginal_likelihood.as_mut().unwrap();
    diagnostics.log_marginal_likelihood = None;
    diagnostics.n2ll = None;
    diagnostics.n2ll_mcse = None;
    let missing = diagnostics.subjects.get_mut(1).expect("second subject");
    missing.mode.clear();
    missing.mode_converged = None;
    clear_subject_n2_numerics(missing);
    missing.failure = Some(MarginalLikelihoodFailureReason::MissingConditionalMode);
    synchronize_unavailable_n2_tables(&mut record);
    write_record(&path, &record);

    let round_trip =
        ParametricResultRecord::read_json(&path).expect("non-first missing mode should round trip");
    let diagnostics = round_trip.marginal_likelihood.unwrap();
    assert_eq!(
        diagnostics.status,
        MarginalLikelihoodStatus::Unavailable {
            failures: vec![MarginalLikelihoodSubjectFailure {
                subject_id: "s2".to_string(),
                reason: MarginalLikelihoodFailureReason::MissingConditionalMode,
            }]
        }
    );
    assert!(diagnostics.log_marginal_likelihood.is_none());
    assert!(diagnostics.n2ll.is_none());
    assert!(diagnostics.n2ll_mcse.is_none());
    assert_eq!(diagnostics.subjects[0].mode_converged, Some(true));
    assert_eq!(diagnostics.subjects[0].mode.len(), 1);
    assert!(diagnostics.subjects[0].failure.is_none());
    let missing = &diagnostics.subjects[1];
    assert_eq!(missing.subject_id, "s2");
    assert_eq!(missing.dimension, 1);
    assert_eq!(missing.samples, 1024);
    assert_eq!(
        missing.seed,
        Some(pmcore::estimation::parametric::marginal_likelihood_subject_seed(8352, 1,))
    );
    assert!(missing.mode.is_empty());
    assert_eq!(missing.mode_converged, None);
    assert_eq!(
        missing.failure,
        Some(MarginalLikelihoodFailureReason::MissingConditionalMode)
    );
    assert!(missing.log_marginal_likelihood.is_none());
    assert!(missing.n2ll.is_none());
    assert!(missing.effective_sample_size.is_none());
    assert!(missing.effective_sample_fraction.is_none());
    assert!(missing.var_log.is_none());
    assert!(missing.n2ll_mcse.is_none());
    fs::remove_file(path).expect("remove missing mode fixture");
}

#[test]
fn schema_six_rejects_absent_or_fabricated_mode_metadata_for_other_failures() {
    let result = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8354)))
        .expect("mode metadata validation fixture");
    let path = std::env::temp_dir().join(format!(
        "pmcore-n2-invalid-mode-metadata-{}.json",
        std::process::id()
    ));
    result.write_json(&path, 0.0, 0.0).expect("write fixture");
    let original = ParametricResultRecord::read_json(&path).expect("read fixture");

    let mut absent_finite_mode_status = original.clone();
    let diagnostics = absent_finite_mode_status
        .marginal_likelihood
        .as_mut()
        .unwrap();
    diagnostics.log_marginal_likelihood = None;
    diagnostics.n2ll = None;
    diagnostics.n2ll_mcse = None;
    let failed = &mut diagnostics.subjects[1];
    failed.mode_converged = None;
    clear_subject_n2_numerics(failed);
    failed.failure = Some(MarginalLikelihoodFailureReason::ScoringFailure(
        "finite mode scoring fixture".to_string(),
    ));
    synchronize_unavailable_n2_tables(&mut absent_finite_mode_status);
    write_record(&path, &absent_finite_mode_status);
    let error = ParametricResultRecord::read_json(&path)
        .expect_err("a finite mode failure must retain convergence status");
    assert!(format!("{error:#}").contains("inconsistent proposal metadata"));

    let mut fabricated_missing_mode = original;
    let diagnostics = fabricated_missing_mode
        .marginal_likelihood
        .as_mut()
        .unwrap();
    diagnostics.log_marginal_likelihood = None;
    diagnostics.n2ll = None;
    diagnostics.n2ll_mcse = None;
    let failed = &mut diagnostics.subjects[1];
    failed.mode_converged = None;
    clear_subject_n2_numerics(failed);
    failed.failure = Some(MarginalLikelihoodFailureReason::MissingConditionalMode);
    assert!(!failed.mode.is_empty());
    synchronize_unavailable_n2_tables(&mut fabricated_missing_mode);
    write_record(&path, &fabricated_missing_mode);
    let error = ParametricResultRecord::read_json(&path)
        .expect_err("a missing-mode failure must not fabricate coordinates");
    assert!(format!("{error:#}").contains("inconsistent proposal metadata"));
    fs::remove_file(path).expect("remove mode metadata fixture");
}

#[test]
fn schema_six_rejects_each_invalid_retained_n2_configuration() {
    let result = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8353)))
        .expect("invalid retained config fixture");
    let path = std::env::temp_dir().join(format!(
        "pmcore-n2-invalid-retained-config-{}.json",
        std::process::id()
    ));
    result.write_json(&path, 0.0, 0.0).expect("write fixture");
    let original = ParametricResultRecord::read_json(&path).expect("read fixture");

    for (label, config, expected) in [
        (
            "samples",
            MarginalLikelihoodConfig::new(1, 8353, 5, 1.5),
            "N2 samples_per_subject must be at least 2",
        ),
        (
            "degrees-of-freedom",
            MarginalLikelihoodConfig::new(1024, 8353, 2, 1.5),
            "N2 degrees_of_freedom must be at least 3",
        ),
        (
            "zero-scale",
            MarginalLikelihoodConfig::new(1024, 8353, 5, 0.0),
            "N2 covariance_scale_multiplier must be finite and positive",
        ),
        (
            "negative-scale",
            MarginalLikelihoodConfig::new(1024, 8353, 5, -1.0),
            "N2 covariance_scale_multiplier must be finite and positive",
        ),
    ] {
        let mut malformed = original.clone();
        coordinate_retained_n2_config(&mut malformed, config);
        write_record(&path, &malformed);
        let error = ParametricResultRecord::read_json(&path)
            .expect_err(&format!("{label} config must be rejected"));
        let message = format!("{error:#}");
        assert!(
            message.contains("invalid retained SAEM configuration") && message.contains(expected),
            "{label} should fail for its retained configuration, got: {message}"
        );
    }
    fs::remove_file(path).expect("remove invalid retained config fixture");
}

#[test]
fn schema_six_rejects_missing_malformed_and_reordered_n2_diagnostics() {
    let result = latent_problem()
        .fit_with(fit_config().marginal_likelihood(n2(8401)))
        .expect("N2 persistence fixture");
    let path =
        std::env::temp_dir().join(format!("pmcore-n2-malformed-{}.json", std::process::id()));
    result.write_json(&path, 0.0, 0.0).expect("write N2 JSON");
    let original: serde_json::Value =
        serde_json::from_reader(fs::File::open(&path).expect("open N2 JSON"))
            .expect("parse N2 JSON");

    let mut missing = original.clone();
    missing
        .as_object_mut()
        .unwrap()
        .remove("marginal_likelihood");
    fs::write(&path, serde_json::to_vec_pretty(&missing).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut missing_nested = original.clone();
    missing_nested["config"]
        .as_object_mut()
        .unwrap()
        .remove("marginal_likelihood");
    fs::write(&path, serde_json::to_vec_pretty(&missing_nested).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut inconsistent = original.clone();
    inconsistent["marginal_likelihood"]["n2ll"] = serde_json::Value::Null;
    fs::write(&path, serde_json::to_vec_pretty(&inconsistent).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut malformed_mode = original.clone();
    malformed_mode["marginal_likelihood"]["subjects"][0]["mode"] = serde_json::json!([null]);
    fs::write(&path, serde_json::to_vec_pretty(&malformed_mode).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut invalid_algebra = original.clone();
    invalid_algebra["marginal_likelihood"]["subjects"][0]["n2ll"] = serde_json::json!(123.0);
    fs::write(&path, serde_json::to_vec_pretty(&invalid_algebra).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut invalid_table = original.clone();
    invalid_table["tables"]["marginal_likelihood"][0]["n2ll"] = serde_json::json!(123.0);
    fs::write(&path, serde_json::to_vec_pretty(&invalid_table).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut invalid_statistics = original.clone();
    let marginal_stat = invalid_statistics["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["kind"] == "marginal_likelihood")
        .unwrap();
    marginal_stat["value"] = serde_json::json!(123.0);
    fs::write(
        &path,
        serde_json::to_vec_pretty(&invalid_statistics).unwrap(),
    )
    .unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut reordered = original.clone();
    reordered["marginal_likelihood"]["subjects"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    fs::write(&path, serde_json::to_vec_pretty(&reordered).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());

    let mut old_schema = original;
    old_schema["schema_version"] = serde_json::json!(4);
    fs::write(&path, serde_json::to_vec_pretty(&old_schema).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&path).is_err());
    fs::remove_file(path).expect("remove malformed N2 JSON");
}
