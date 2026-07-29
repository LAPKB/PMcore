use super::*;

pub(super) fn matrix_from_rows(values: &[Vec<f64>], width: usize) -> Result<Array2<f64>> {
    if values.len() != width || values.iter().any(|row| row.len() != width) {
        anyhow::bail!("matrix coordinate width mismatch");
    }
    Ok(Array2::from_shape_vec(
        (width, width),
        values.iter().flatten().copied().collect(),
    )?)
}

pub(super) fn normal_two_sided_z(p: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let norm = Normal::new(0.0, 1.0).expect("standard normal parameters are valid");
    let one_sided = p + (1.0 - p) / 2.0;
    norm.inverse_cdf(one_sided)
}

/// Evaluate one operational convergence criterion.
pub(super) fn evaluate_criterion(
    name: &str,
    observed: Option<f64>,
    threshold: f64,
    predicate: impl FnOnce(f64) -> bool,
) -> OperationalConvergenceCriterion {
    let status = match observed {
        Some(value) if value.is_finite() && predicate(value) => {
            OperationalConvergenceCriterionStatus::Satisfied
        }
        Some(value) if value.is_finite() => OperationalConvergenceCriterionStatus::NotSatisfied,
        Some(_) => OperationalConvergenceCriterionStatus::Unavailable(
            "observed value is non-finite".to_string(),
        ),
        None => OperationalConvergenceCriterionStatus::Unavailable(
            "criterion could not be evaluated".to_string(),
        ),
    };
    OperationalConvergenceCriterion {
        name: name.to_string(),
        observed,
        threshold,
        status,
    }
}

pub(super) fn operational_free_coordinates(
    information: &InformationDiagnostics,
    average: &SaemIterateAverage,
) -> Result<Vec<f64>> {
    information
        .coordinates
        .iter()
        .map(|coordinate| match &coordinate.kind {
            InformationCoordinateKind::Population { parameter_index } => average
                .population_phi
                .get(*parameter_index)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("population coordinate out of range")),
            InformationCoordinateKind::CovariateEffect { effect_index } => average
                .covariate_betas
                .as_ref()
                .and_then(|values| values.get(*effect_index))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("covariate coordinate out of range")),
            InformationCoordinateKind::Omega { row, column } => average
                .omega
                .get((*row, *column))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Omega coordinate out of range")),
            InformationCoordinateKind::OmegaIov { row, column } => average
                .omega_iov
                .as_ref()
                .and_then(|matrix| matrix.get((*row, *column)))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Omega_IOV coordinate out of range")),
            InformationCoordinateKind::Residual {
                output_index,
                component,
            } => {
                let model = average
                    .residual_models
                    .iter()
                    .find(|(index, _)| index == output_index)
                    .map(|(_, model)| model)
                    .ok_or_else(|| anyhow::anyhow!("residual coordinate output unavailable"))?;
                match (model, component.as_str()) {
                    (ResidualErrorModel::Constant { a }, "sigma") => Ok(*a),
                    (ResidualErrorModel::Exponential { sigma }, "sigma") => Ok(*sigma),
                    (ResidualErrorModel::Proportional { b }, "proportional") => Ok(*b),
                    (ResidualErrorModel::Combined { a, .. }, "additive")
                    | (ResidualErrorModel::CorrelatedCombined { a, .. }, "additive") => Ok(*a),
                    (ResidualErrorModel::Combined { b, .. }, "proportional")
                    | (ResidualErrorModel::CorrelatedCombined { b, .. }, "proportional") => Ok(*b),
                    (ResidualErrorModel::CorrelatedCombined { rho, .. }, "correlation") => Ok(*rho),
                    _ => anyhow::bail!("residual coordinate component mismatch"),
                }
            }
        })
        .collect()
}

pub(super) fn operational_simulation_sd_fraction(
    information: &InformationDiagnostics,
    markov: &MarkovSimulationVarianceDiagnostics,
) -> Option<f64> {
    let width = information.coordinates.len();
    let observed = matrix_from_rows(&information.observed_information, width).ok()?;
    let covariance = matrix_from_rows(&markov.simulation_covariance, width).ok()?;
    worst_contrast(&observed, &covariance).ok()
}

pub(super) fn solve_spd(matrix: &Array2<f64>, rhs: &[f64]) -> Option<Vec<f64>> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != rhs.len() {
        return None;
    }
    let lower = cholesky_lower(matrix).ok()?;
    let n = rhs.len();
    let mut y = vec![0.0; n];
    for row in 0..n {
        let subtotal = (0..row)
            .map(|column| lower[row][column] * y[column])
            .sum::<f64>();
        y[row] = (rhs[row] - subtotal) / lower[row][row];
    }
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let subtotal = ((row + 1)..n)
            .map(|column| lower[column][row] * result[column])
            .sum::<f64>();
        result[row] = (y[row] - subtotal) / lower[row][row];
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

/// Invariant Newton displacement `sqrt(g^T Iobs^-1 g)`.
pub(super) fn newton_displacement(
    info: &InformationDiagnostics,
    markov: &MarkovSimulationVarianceDiagnostics,
) -> Option<f64> {
    let width = info.coordinates.len();
    if width == 0 || markov.grand_score_mean.len() != width {
        return None;
    }
    let observed = matrix_from_rows(&info.observed_information, width).ok()?;
    let displacement = solve_spd(&observed, &markov.grand_score_mean)?;
    let squared = markov
        .grand_score_mean
        .iter()
        .zip(&displacement)
        .map(|(score, step)| score * step)
        .sum::<f64>();
    (squared.is_finite() && squared >= 0.0).then(|| squared.sqrt())
}

/// Worst-direction Newton-step MC SD from diagnostic-mean LRV/draws.
pub(super) fn newton_displacement_mc_sd(
    info: &InformationDiagnostics,
    markov: &MarkovSimulationVarianceDiagnostics,
) -> Option<f64> {
    let width = info.coordinates.len();
    let draws = markov.config?.draws_per_chain;
    if width == 0 || draws == 0 {
        return None;
    }
    let observed = matrix_from_rows(&info.observed_information, width).ok()?;
    let mut score_covariance =
        matrix_from_rows(markov.rank_diagnostics.diagnostic_mean_lrv.as_ref()?, width).ok()?;
    score_covariance /= draws as f64;
    let mut inverse = Array2::zeros((width, width));
    for column in 0..width {
        let mut unit = vec![0.0; width];
        unit[column] = 1.0;
        let solved = solve_spd(&observed, &unit)?;
        for row in 0..width {
            inverse[[row, column]] = solved[row];
        }
    }
    let mut mapped = Array2::zeros((width, width));
    for row in 0..width {
        for column in 0..=row {
            let mut value = 0.0;
            for left in 0..width {
                for right in 0..width {
                    value += inverse[[row, left]]
                        * score_covariance[[left, right]]
                        * inverse[[column, right]];
                }
            }
            mapped[[row, column]] = value;
            mapped[[column, row]] = value;
        }
    }
    worst_contrast(&observed, &mapped).ok()
}

pub(super) fn incremental_average(previous: f64, current: f64, count: usize) -> f64 {
    previous + (current - previous) / count as f64
}

pub(super) fn average_covariance(
    average: &mut Array2<f64>,
    current: &Array2<f64>,
    estimated_mask: &Array2<bool>,
    count: usize,
) {
    for row in 0..average.nrows() {
        for col in 0..=row {
            if estimated_mask[[row, col]] {
                let value = incremental_average(average[[row, col]], current[[row, col]], count);
                average[[row, col]] = value;
                average[[col, row]] = value;
            }
        }
    }
}

pub(super) fn average_residual_model(
    previous: ResidualErrorModel,
    current: ResidualErrorModel,
    estimated: bool,
    components: [bool; 2],
    correlated_components: [bool; 3],
    count: usize,
) -> Result<ResidualErrorModel> {
    let averaged = match (previous, current) {
        (ResidualErrorModel::Constant { a }, ResidualErrorModel::Constant { a: current }) => {
            ResidualErrorModel::Constant {
                a: if estimated {
                    incremental_average(a, current, count)
                } else {
                    a
                },
            }
        }
        (
            ResidualErrorModel::Proportional { b },
            ResidualErrorModel::Proportional { b: current },
        ) => ResidualErrorModel::Proportional {
            b: if estimated {
                incremental_average(b, current, count)
            } else {
                b
            },
        },
        (
            ResidualErrorModel::Exponential { sigma },
            ResidualErrorModel::Exponential { sigma: current },
        ) => ResidualErrorModel::Exponential {
            sigma: if estimated {
                incremental_average(sigma, current, count)
            } else {
                sigma
            },
        },
        (
            ResidualErrorModel::Combined { a, b },
            ResidualErrorModel::Combined {
                a: current_a,
                b: current_b,
            },
        ) => ResidualErrorModel::Combined {
            a: if components[0] {
                incremental_average(a, current_a, count)
            } else {
                a
            },
            b: if components[1] {
                incremental_average(b, current_b, count)
            } else {
                b
            },
        },
        (
            ResidualErrorModel::CorrelatedCombined { a, b, rho },
            ResidualErrorModel::CorrelatedCombined {
                a: current_a,
                b: current_b,
                rho: current_rho,
            },
        ) => ResidualErrorModel::CorrelatedCombined {
            a: if correlated_components[0] {
                incremental_average(a, current_a, count)
            } else {
                a
            },
            b: if correlated_components[1] {
                incremental_average(b, current_b, count)
            } else {
                b
            },
            rho: if correlated_components[2] {
                incremental_average(rho, current_rho, count)
            } else {
                rho
            },
        },
        _ => anyhow::bail!("residual family changed while accumulating SAEM averages"),
    };
    Ok(averaged)
}

pub(super) fn validate_average_population(
    values: &[f64],
    initialization: &SaemInitialization,
) -> Result<()> {
    let initial = population_phi(
        &initialization.initial_population_parameters,
        &initialization.parameter_scales,
    )?;
    if values.len() != initial.len() || values.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("averaged population phi values must be finite and retain their width");
    }
    for index in 0..values.len() {
        if !initialization.estimated_parameters[index] && values[index] != initial[index] {
            anyhow::bail!("averaged population phi changed fixed coordinate {index}");
        }
    }
    Ok(())
}

pub(super) fn validate_average_covariance(
    matrix: &Array2<f64>,
    specification: &ResolvedOmega,
    label: &str,
) -> Result<()> {
    if matrix.raw_dim() != specification.initial().raw_dim() {
        anyhow::bail!("averaged {label} has an invalid shape");
    }
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[[row, col]];
            if !value.is_finite() || value != matrix[[col, row]] {
                anyhow::bail!("averaged {label} must be finite and symmetric");
            }
            if !specification.structural_mask()[[row, col]] && value != 0.0 {
                anyhow::bail!("averaged {label} changed a structural zero");
            }
            if !specification.estimated_mask()[[row, col]]
                && value != specification.initial()[[row, col]]
            {
                anyhow::bail!("averaged {label} changed a fixed entry");
            }
        }
    }
    cholesky_lower(matrix)
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("averaged {label} is not positive definite: {error}"))
}

pub(super) fn validate_average_residuals(
    original_width: usize,
    models: &[(usize, ResidualErrorModel)],
    declarations: &ParametricErrorModels,
) -> Result<()> {
    if original_width != declarations.models().len()
        || models.len() != declarations.models().iter().count()
    {
        anyhow::bail!("averaged residual output collection changed");
    }
    for ((output, model), (declared_output, terminal)) in models.iter().copied().zip(
        declarations
            .models()
            .iter()
            .map(|(index, model)| (index, *model)),
    ) {
        if output != declared_output || output >= original_width {
            anyhow::bail!("averaged residual output indices changed");
        }
        let output_name = declarations
            .output_name(output)
            .ok_or_else(|| anyhow::anyhow!("averaged residual output {output} has no name"))?;
        let components = declarations.combined_component_estimated(output);
        if !declarations.is_estimated(output) && model != terminal {
            anyhow::bail!(
                "averaged residual model changed fixed output '{output_name}' at index {output}"
            );
        }
        if let (
            ResidualErrorModel::Combined { a, b },
            ResidualErrorModel::Combined {
                a: terminal_a,
                b: terminal_b,
            },
        ) = (model, terminal)
        {
            if (!components[0] && a != terminal_a) || (!components[1] && b != terminal_b) {
                anyhow::bail!(
                    "averaged residual model changed a fixed component for output '{output_name}' at index {output}"
                );
            }
        }
        let correlated_components = declarations.correlated_combined_component_estimated(output);
        if let (
            ResidualErrorModel::CorrelatedCombined { a, b, rho },
            ResidualErrorModel::CorrelatedCombined {
                a: terminal_a,
                b: terminal_b,
                rho: terminal_rho,
            },
        ) = (model, terminal)
        {
            if (!correlated_components[0] && a != terminal_a)
                || (!correlated_components[1] && b != terminal_b)
                || (!correlated_components[2] && rho != terminal_rho)
            {
                anyhow::bail!(
                    "averaged correlated-combined model changed a fixed component for output '{output_name}' at index {output}"
                );
            }
        }
        let valid = match model {
            ResidualErrorModel::Constant { a } => a.is_finite() && a > 0.0,
            ResidualErrorModel::Proportional { b } => b.is_finite() && b > 0.0,
            ResidualErrorModel::Exponential { sigma } => sigma.is_finite() && sigma > 0.0,
            ResidualErrorModel::Combined { a, b } => {
                a.is_finite()
                    && b.is_finite()
                    && a >= 0.0
                    && b >= 0.0
                    && (!components[0] || a > 0.0)
                    && (!components[1] || b > 0.0)
            }
            ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                a.is_finite()
                    && a > 0.0
                    && b.is_finite()
                    && b > 0.0
                    && rho.is_finite()
                    && rho > -1.0
                    && rho < 1.0
            }
        };
        if !valid {
            anyhow::bail!(
                "averaged residual model for output '{output_name}' at index {output} is outside its domain"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct WarningCount {
    first_iteration: Option<usize>,
    cycles: usize,
    count: usize,
}

impl WarningCount {
    fn record_cycle(&mut self, iteration: usize) {
        self.first_iteration.get_or_insert(iteration);
        self.cycles += 1;
    }

    fn record_count(&mut self, iteration: usize, count: usize) {
        if count == 0 {
            return;
        }
        self.first_iteration.get_or_insert(iteration);
        self.count += count;
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct CovarianceBoundaryRejectionSummary {
    pub(super) first_iteration: Option<usize>,
    pub(super) longest_run: usize,
}

pub(super) fn covariance_boundary_rejection_summary(
    cycles: &[SaemCycleDiagnostics],
    policy: CovarianceStabilityConfig,
    iov: bool,
) -> CovarianceBoundaryRejectionSummary {
    let mut summary = CovarianceBoundaryRejectionSummary::default();
    let mut current_run = 0usize;
    let mut current_start = None;
    for cycle in cycles {
        let (rejected, margin) = if iov {
            (
                cycle.omega_iov_update_rejected,
                cycle.omega_iov_relative_spd_margin,
            )
        } else {
            (cycle.omega_update_rejected, cycle.omega_relative_spd_margin)
        };
        if rejected && margin.is_some_and(|value| value <= policy.minimum_relative_spd_margin) {
            if current_run == 0 {
                current_start = Some(cycle.iteration);
            }
            current_run += 1;
            summary.longest_run = summary.longest_run.max(current_run);
            if current_run >= policy.rejection_window && summary.first_iteration.is_none() {
                summary.first_iteration = current_start;
            }
        } else {
            current_run = 0;
            current_start = None;
        }
    }
    summary
}

pub(super) fn parametric_warnings(
    cycles: &[SaemCycleDiagnostics],
    covariance_stability: Option<CovarianceStabilityConfig>,
) -> Vec<ParametricWarning> {
    let mut omega = WarningCount::default();
    let mut omega_iov = WarningCount::default();
    let mut eta_non_finite = WarningCount::default();
    let mut eta_block_non_finite = WarningCount::default();
    let mut kappa_non_finite = WarningCount::default();
    let mut residual_rejected = BTreeMap::<String, WarningCount>::new();
    let mut proportional_floor = BTreeMap::<String, WarningCount>::new();
    let mut residual_non_finite = BTreeMap::<String, WarningCount>::new();
    let mut exponential_domain = BTreeMap::<String, WarningCount>::new();
    let mut additive_collapse = BTreeMap::<String, WarningCount>::new();
    let mut optimizer_not_converged = BTreeMap::<String, WarningCount>::new();

    for cycle in cycles {
        if cycle.omega_update_rejected {
            omega.record_cycle(cycle.iteration);
        }
        if cycle.omega_iov_update_rejected {
            omega_iov.record_cycle(cycle.iteration);
        }
        eta_non_finite.record_count(cycle.iteration, cycle.eta_non_finite);
        eta_block_non_finite.record_count(cycle.iteration, cycle.eta_block_non_finite);
        kappa_non_finite.record_count(cycle.iteration, cycle.kappa_non_finite);
        for residual in &cycle.residual_diagnostics {
            if residual.update_rejected {
                residual_rejected
                    .entry(residual.output.clone())
                    .or_default()
                    .record_cycle(cycle.iteration);
            }
            proportional_floor
                .entry(residual.output.clone())
                .or_default()
                .record_count(cycle.iteration, residual.proportional_floor_count);
            residual_non_finite
                .entry(residual.output.clone())
                .or_default()
                .record_count(cycle.iteration, residual.non_finite_prediction_count);
            exponential_domain
                .entry(residual.output.clone())
                .or_default()
                .record_count(cycle.iteration, residual.exponential_domain_violation_count);
            if residual.combined_additive_collapse_warning {
                additive_collapse
                    .entry(residual.output.clone())
                    .or_default()
                    .record_cycle(cycle.iteration);
            }
            if residual.optimizer_converged == Some(false) {
                optimizer_not_converged
                    .entry(residual.output.clone())
                    .or_default()
                    .record_cycle(cycle.iteration);
            }
        }
    }

    let mut warnings = Vec::new();
    if let Some(first_iteration) = omega.first_iteration {
        warnings.push(ParametricWarning::OmegaUpdateRejected {
            first_iteration,
            cycles: omega.cycles,
        });
    }
    if let Some(first_iteration) = omega_iov.first_iteration {
        warnings.push(ParametricWarning::OmegaIovUpdateRejected {
            first_iteration,
            cycles: omega_iov.cycles,
        });
    }
    if let Some(policy) = covariance_stability {
        let omega_boundary = covariance_boundary_rejection_summary(cycles, policy, false);
        if let Some(first_iteration) = omega_boundary.first_iteration {
            warnings.push(ParametricWarning::OmegaBoundaryRejection {
                first_iteration,
                longest_run: omega_boundary.longest_run,
            });
        }
        let omega_iov_boundary = covariance_boundary_rejection_summary(cycles, policy, true);
        if let Some(first_iteration) = omega_iov_boundary.first_iteration {
            warnings.push(ParametricWarning::OmegaIovBoundaryRejection {
                first_iteration,
                longest_run: omega_iov_boundary.longest_run,
            });
        }
    }
    if let Some(first_iteration) = eta_non_finite.first_iteration {
        warnings.push(ParametricWarning::EtaNonFiniteProposals {
            first_iteration,
            count: eta_non_finite.count,
        });
    }
    if let Some(first_iteration) = eta_block_non_finite.first_iteration {
        warnings.push(ParametricWarning::EtaBlockNonFiniteProposals {
            first_iteration,
            count: eta_block_non_finite.count,
        });
    }
    if let Some(first_iteration) = kappa_non_finite.first_iteration {
        warnings.push(ParametricWarning::KappaNonFiniteProposals {
            first_iteration,
            count: kappa_non_finite.count,
        });
    }
    for (output, warning) in residual_rejected {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ResidualUpdateRejected {
                output,
                first_iteration,
                cycles: warning.cycles,
            });
        }
    }
    for (output, warning) in proportional_floor {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ProportionalPredictionFloor {
                output,
                first_iteration,
                count: warning.count,
            });
        }
    }
    for (output, warning) in residual_non_finite {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::NonFiniteResidualPrediction {
                output,
                first_iteration,
                count: warning.count,
            });
        }
    }
    for (output, warning) in exponential_domain {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ExponentialDomainViolation {
                output,
                first_iteration,
                count: warning.count,
            });
        }
    }
    for (output, warning) in additive_collapse {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::CombinedAdditiveCollapse {
                output,
                first_iteration,
                cycles: warning.cycles,
            });
        }
    }
    for (output, warning) in optimizer_not_converged {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ResidualOptimizerNotConverged {
                output,
                first_iteration,
                cycles: warning.cycles,
            });
        }
    }
    warnings
}

pub(super) fn calculate_result_marginal_likelihood<E: Equation>(
    state: &SaemState<E>,
    conditional_modes: &[SubjectConditionalMode],
    conditional_mode_error: Option<&str>,
) -> Option<MarginalLikelihoodDiagnostics> {
    let config = state.config.marginal_likelihood?;
    let n_eta = state.initialization.random_effect_indices.len();
    let n_kappa = state.initialization.iov_effect_indices.len();
    let latent = n_eta > 0 || n_kappa > 0;
    let occasion_indices = state
        .data
        .subjects()
        .iter()
        .map(|subject| {
            if n_kappa == 0 {
                Vec::new()
            } else {
                subject
                    .occasions()
                    .iter()
                    .map(|occasion| occasion.index())
                    .collect()
            }
        })
        .collect::<Vec<Vec<usize>>>();
    let mut flattened_modes = Vec::with_capacity(state.initialization.subject_ids.len());
    let mut converged = Vec::with_capacity(state.initialization.subject_ids.len());
    let mut validation_failures = Vec::with_capacity(state.initialization.subject_ids.len());

    for (subject_index, subject_id) in state.initialization.subject_ids.iter().enumerate() {
        if !latent {
            flattened_modes.push(Vec::new());
            converged.push(None);
            validation_failures.push(None);
            continue;
        }
        let Some(mode) = conditional_modes.get(subject_index) else {
            flattened_modes.push(Vec::new());
            converged.push(None);
            validation_failures.push(Some(
                MarginalLikelihoodFailureReason::MissingConditionalMode,
            ));
            continue;
        };
        let mut validation_failure = None;
        if mode.subject_id != *subject_id {
            validation_failure.get_or_insert(MarginalLikelihoodFailureReason::SubjectIdMismatch {
                expected: subject_id.clone(),
                actual: mode.subject_id.clone(),
            });
        }
        if mode.eta.len() != n_eta {
            validation_failure.get_or_insert(MarginalLikelihoodFailureReason::EtaWidthMismatch {
                expected: n_eta,
                actual: mode.eta.len(),
            });
        }
        if mode.kappas.len() != occasion_indices[subject_index].len() {
            validation_failure.get_or_insert(MarginalLikelihoodFailureReason::KappaCountMismatch {
                expected: occasion_indices[subject_index].len(),
                actual: mode.kappas.len(),
            });
        }
        for (position, kappa) in mode.kappas.iter().enumerate() {
            if let Some(expected) = occasion_indices[subject_index].get(position) {
                if kappa.occasion_index != *expected {
                    validation_failure.get_or_insert(
                        MarginalLikelihoodFailureReason::KappaOccasionMismatch {
                            position,
                            expected: *expected,
                            actual: kappa.occasion_index,
                        },
                    );
                }
            }
            if kappa.values.len() != n_kappa {
                validation_failure.get_or_insert(
                    MarginalLikelihoodFailureReason::KappaWidthMismatch {
                        position,
                        expected: n_kappa,
                        actual: kappa.values.len(),
                    },
                );
            }
        }
        let mut flattened = mode.eta.clone();
        for kappa in &mode.kappas {
            flattened.extend_from_slice(&kappa.values);
        }
        if flattened.iter().any(|value| !value.is_finite()) {
            validation_failure
                .get_or_insert(MarginalLikelihoodFailureReason::NonFiniteModeCoordinate);
        }
        flattened_modes.push(flattened);
        converged.push(Some(mode.converged));
        validation_failures.push(validation_failure);
    }

    let curvature_covariances = conditional_modes
        .iter()
        .map(|mode| {
            mode.uncertainty
                .latent_covariance
                .as_ref()
                .and_then(|rows| matrix_from_rows(rows, rows.len()).ok())
        })
        .collect::<Vec<_>>();
    let subjects = state
        .initialization
        .subject_ids
        .iter()
        .enumerate()
        .map(|(index, subject_id)| MarginalSubject {
            subject_id,
            occasion_indices: &occasion_indices[index],
            mode: &flattened_modes[index],
            mode_converged: converged[index],
            eta_dimension: n_eta,
            kappa_dimension: n_kappa,
            validation_failure: validation_failures[index].clone(),
            curvature_availability: conditional_modes
                .get(index)
                .map(|mode| &mode.uncertainty.status),
            curvature_covariance: curvature_covariances.get(index).and_then(Option::as_ref),
        })
        .collect::<Vec<_>>();
    if let Some(error) = conditional_mode_error {
        return Some(unavailable_population_marginal_likelihood(
            config,
            &subjects,
            MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(format!(
                "global conditional mode calculation failed: {error}"
            )),
        ));
    }
    Some(calculate_population_marginal_likelihood(
        config,
        &subjects,
        &state.omega,
        state.omega_iov.as_ref(),
        |subject_index, eta, kappas| {
            state
                .score_subject_latents(subject_index, eta, kappas)
                .map(SubjectPosteriorScore::log_posterior)
        },
    ))
}

pub(super) fn conditional_modes<E: Equation>(
    state: &SaemState<E>,
) -> Result<Vec<SubjectConditionalMode>> {
    if !state.compute_map {
        return Ok(Vec::new());
    }

    let n_eta = state.initialization.random_effect_indices.len();
    let n_kappa = state.initialization.iov_effect_indices.len();
    if n_eta == 0 && n_kappa == 0 {
        return Ok(Vec::new());
    }
    let mut modes = Vec::with_capacity(state.initialization.subject_ids.len());
    for (subject_index, subject_id) in state.initialization.subject_ids.iter().enumerate() {
        let eta_start = mean_vectors(state.etas[subject_index].iter().map(|eta| eta.as_slice()))?;
        let occasion_count = if state.omega_iov.is_some() {
            state.data.subjects()[subject_index].occasions().len()
        } else {
            0
        };
        let mut kappa_start = Vec::with_capacity(occasion_count);
        for occasion_position in 0..occasion_count {
            kappa_start.push(mean_vectors(
                state.kappas[subject_index]
                    .iter()
                    .map(|chain| chain[occasion_position].as_slice()),
            )?);
        }
        let mut initial = eta_start;
        for kappa in &kappa_start {
            initial.extend_from_slice(kappa);
        }

        let step_fraction = state.map_initial_step;
        let mut scales = (0..n_eta)
            .map(|index| state.omega[[index, index]].sqrt() * step_fraction)
            .collect::<Vec<_>>();
        if let Some(omega_iov) = state.omega_iov.as_ref() {
            for _ in 0..occasion_count {
                scales.extend(
                    (0..n_kappa).map(|index| omega_iov[[index, index]].sqrt() * step_fraction),
                );
            }
        }
        for scale in &mut scales {
            *scale = scale.max(1e-6);
        }

        let solution = optimize_conditional_mode(
            initial,
            &scales,
            state.map_max_iterations as u64,
            state.map_sd_tolerance,
            |coordinates| {
                let (eta, kappas) = unflatten_latents(coordinates, n_eta, occasion_count, n_kappa);
                match state.score_subject_latents(subject_index, eta, &kappas) {
                    Ok(score) if score.log_posterior().is_finite() => -score.log_posterior(),
                    _ => f64::INFINITY,
                }
            },
        )?;
        let mut coordinates = (0..n_eta)
            .map(|index| JointLatentCoordinate {
                index,
                name: format!("eta:{}", state.initialization.random_effect_names[index]),
                kind: JointLatentCoordinateKind::Eta {
                    parameter_index: state.initialization.random_effect_indices[index],
                },
                prior_sd: state.omega[[index, index]].sqrt(),
            })
            .collect::<Vec<_>>();
        if let Some(omega_iov) = state.omega_iov.as_ref() {
            for occasion_position in 0..occasion_count {
                let occasion_index =
                    state.data.subjects()[subject_index].occasions()[occasion_position].index();
                for effect_index in 0..n_kappa {
                    coordinates.push(JointLatentCoordinate {
                        index: n_eta + occasion_position * n_kappa + effect_index,
                        name: format!(
                            "kappa:{occasion_index}:{}",
                            state.initialization.iov_effect_names[effect_index]
                        ),
                        kind: JointLatentCoordinateKind::Kappa {
                            occasion_index,
                            effect_index,
                            parameter_index: state.initialization.iov_effect_indices[effect_index],
                        },
                        prior_sd: omega_iov[[effect_index, effect_index]].sqrt(),
                    });
                }
            }
        }
        let prior_sds = coordinates
            .iter()
            .map(|coordinate| coordinate.prior_sd)
            .collect::<Vec<_>>();
        let mode_metadata = ConditionalModeMetadata {
            converged: solution.converged,
            iterations: solution.iterations,
            objective_value: solution.objective,
            termination_message: solution.termination.clone(),
        };
        let uncertainty = conditional_mode_curvature(
            &solution.coordinates,
            &prior_sds,
            &coordinates,
            &mode_metadata,
            |coordinates| {
                let (eta, kappas) = unflatten_latents(coordinates, n_eta, occasion_count, n_kappa);
                match state.score_subject_latents(subject_index, eta, &kappas) {
                    Ok(score) if score.log_posterior().is_finite() => -score.log_posterior(),
                    _ => f64::INFINITY,
                }
            },
        );
        let (eta, kappas) =
            unflatten_latents(&solution.coordinates, n_eta, occasion_count, n_kappa);
        let parameters = state.individual_parameters_from_eta(subject_index, eta)?;
        let kappa_estimates = kappas
            .into_iter()
            .enumerate()
            .map(|(occasion_position, values)| OccasionKappaEstimate {
                subject_id: subject_id.clone(),
                occasion_index: state.data.subjects()[subject_index].occasions()[occasion_position]
                    .index(),
                values,
            })
            .collect();
        modes.push(SubjectConditionalMode {
            subject_id: subject_id.clone(),
            eta: eta.to_vec(),
            kappas: kappa_estimates,
            parameters,
            objective: solution.objective,
            converged: solution.converged,
            iterations: solution.iterations,
            termination: solution.termination,
            uncertainty,
        });
    }
    Ok(modes)
}

pub(super) fn unflatten_latents(
    coordinates: &[f64],
    n_eta: usize,
    occasion_count: usize,
    n_kappa: usize,
) -> (&[f64], Vec<Vec<f64>>) {
    let eta = &coordinates[..n_eta];
    let kappas = (0..occasion_count)
        .map(|occasion| {
            let start = n_eta + occasion * n_kappa;
            coordinates[start..start + n_kappa].to_vec()
        })
        .collect();
    (eta, kappas)
}

pub(super) fn mean_vectors<'a>(vectors: impl IntoIterator<Item = &'a [f64]>) -> Result<Vec<f64>> {
    let mut vectors = vectors.into_iter();
    let Some(first) = vectors.next() else {
        anyhow::bail!("cannot summarize random effects without chains");
    };
    let mut mean = first.to_vec();
    let mut count = 1usize;
    for vector in vectors {
        if vector.len() != mean.len() {
            anyhow::bail!("random-effect chains have inconsistent dimensions");
        }
        for (sum, value) in mean.iter_mut().zip(vector) {
            *sum += value;
        }
        count += 1;
    }
    for value in &mut mean {
        *value /= count as f64;
    }
    Ok(mean)
}

pub(super) fn zero_etas(
    n_subjects: usize,
    n_chains: usize,
    n_parameters: usize,
) -> Vec<Vec<Vec<f64>>> {
    vec![vec![vec![0.0; n_parameters]; n_chains]; n_subjects]
}

pub(super) fn zero_kappas(
    occasion_counts: &[usize],
    n_chains: usize,
    n_kappa: usize,
) -> Vec<Vec<Vec<Vec<f64>>>> {
    occasion_counts
        .iter()
        .map(|&n_occasions| vec![vec![vec![0.0; n_kappa]; n_occasions]; n_chains])
        .collect()
}

pub(super) fn second_moment_from_etas(etas: &[Vec<Vec<f64>>]) -> Result<Array2<f64>> {
    let mut samples = etas.iter().flat_map(|subject_chains| subject_chains.iter());
    let Some(first) = samples.next() else {
        anyhow::bail!("cannot update omega without subject-chain samples");
    };
    let dimension = first.len();
    let mut second_moment = Array2::zeros((dimension, dimension));
    let mut count = 0usize;
    for eta in std::iter::once(first).chain(samples) {
        if eta.len() != dimension {
            anyhow::bail!("eta samples have inconsistent dimensions");
        }
        for row in 0..dimension {
            for col in 0..dimension {
                second_moment[[row, col]] += eta[row] * eta[col];
            }
        }
        count += 1;
    }
    second_moment.mapv_inplace(|value| value / count as f64);
    Ok(second_moment)
}

pub(super) fn covariance_from_kappas(kappas: &[Vec<Vec<Vec<f64>>>]) -> Result<Array2<f64>> {
    let mut samples = kappas
        .iter()
        .flat_map(|subject_chains| subject_chains.iter())
        .flat_map(|chains| chains.iter());
    let Some(first) = samples.next() else {
        anyhow::bail!("cannot update omega_iov without occasion samples");
    };
    let dimension = first.len();
    let mut covariance = Array2::zeros((dimension, dimension));
    let mut count = 0usize;
    for kappa in std::iter::once(first).chain(samples) {
        if kappa.len() != dimension {
            anyhow::bail!("kappa samples have inconsistent dimensions");
        }
        for row in 0..dimension {
            for col in 0..dimension {
                covariance[[row, col]] += kappa[row] * kappa[col];
            }
        }
        count += 1;
    }
    covariance.mapv_inplace(|value| value / count as f64);
    Ok(covariance)
}

pub(super) fn correlated_random_walk(
    current: &[f64],
    lower: &[Vec<f64>],
    standard_normals: &[f64],
    scale: f64,
) -> Result<Vec<f64>> {
    anyhow::ensure!(
        lower.len() == current.len()
            && standard_normals.len() == current.len()
            && lower
                .iter()
                .enumerate()
                .all(|(row, values)| values.len() > row),
        "correlated random-walk dimensions do not match"
    );
    Ok((0..current.len())
        .map(|row| {
            let perturbation = (0..=row)
                .map(|column| lower[row][column] * standard_normals[column])
                .sum::<f64>();
            current[row] + scale * perturbation
        })
        .collect())
}

pub(super) fn initial_proposal_step_sizes(omega: &Array2<f64>, rw_init: f64) -> Vec<f64> {
    (0..omega.nrows())
        .map(|index| omega[[index, index]].max(f64::EPSILON).sqrt() * rw_init)
        .collect()
}

pub(super) fn adapt_component_step_size(current: f64, acceptance_rate: f64) -> f64 {
    adapt_block_step_size(current, acceptance_rate, COMPONENT_TARGET_ACCEPTANCE)
}

pub(super) fn adapt_block_step_size(current: f64, acceptance_rate: f64, target: f64) -> f64 {
    if acceptance_rate > target {
        (current * PROPOSAL_SCALE_INCREASE).min(MAX_PROPOSAL_SCALE)
    } else {
        (current * PROPOSAL_SCALE_DECREASE).max(MIN_PROPOSAL_SCALE)
    }
}

pub(super) fn zero_eta_subject_phi(
    population_parameters: &[f64],
    initialization: &SaemInitialization,
) -> Result<Vec<Vec<f64>>> {
    let phi = population_phi(population_parameters, &initialization.parameter_scales)?;
    Ok(vec![phi; initialization.subject_ids.len()])
}
