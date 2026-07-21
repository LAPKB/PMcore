//! Analytic complete-data derivatives and observed-information recursion.
//!
//! Coordinates are population φ values, raw free covariance entries, and raw
//! positive residual standard-deviation components. No model sensitivities,
//! finite differences, regularization, or matrix repair are used here.

use anyhow::{bail, Result};
use ndarray::Array2;
use pharmsol::prelude::simulator::Prediction;
use pharmsol::{Censor, Predictions};

use crate::estimation::ParametricErrorModels;
use crate::results::{
    InformationCoordinate, InformationCoordinateKind, InformationDiagnostics, InformationStatus,
    PopulationUncertaintyDiagnostics, PopulationUncertaintyRegularization,
    PopulationUncertaintyStatus, PopulationUncertaintyUnavailableReason,
};
use crate::ResidualErrorModel;

use super::covariance::{cholesky_lower, eigenvalue_extrema_symmetric, inverse_spd_from_cholesky};

#[derive(Debug, Clone)]
pub(crate) struct InformationLayout {
    pub(crate) coordinates: Vec<InformationCoordinate>,
    population: Vec<Option<usize>>,
    covariate_effects: Vec<Option<usize>>,
    omega: Vec<CovarianceCoordinate>,
    omega_iov: Vec<CovarianceCoordinate>,
    residual: Vec<ResidualCoordinates>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CovarianceCoordinate {
    coordinate: usize,
    row: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ResidualCoordinates {
    pub(crate) additive: Option<usize>,
    pub(crate) proportional: Option<usize>,
    pub(crate) correlation: Option<usize>,
}

impl InformationLayout {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        parameter_names: &[String],
        estimated_parameters: &[bool],
        covariate_effect_names: &[String],
        covariate_estimated: &[bool],
        random_effect_names: &[String],
        omega_structural: &Array2<bool>,
        omega_estimated: &Array2<bool>,
        iov_effect_names: &[String],
        omega_iov_structural: Option<&Array2<bool>>,
        omega_iov_estimated: Option<&Array2<bool>>,
        error_models: &ParametricErrorModels,
    ) -> Result<Self> {
        if parameter_names.len() != estimated_parameters.len() {
            bail!("population parameter metadata dimension mismatch");
        }
        if covariate_effect_names.len() != covariate_estimated.len() {
            bail!("covariate effect metadata dimension mismatch");
        }
        validate_masks(
            random_effect_names.len(),
            omega_structural,
            omega_estimated,
            "omega",
        )?;
        match (omega_iov_structural, omega_iov_estimated) {
            (Some(structural), Some(estimated)) => {
                validate_masks(iov_effect_names.len(), structural, estimated, "omega_iov")?;
            }
            (None, None) if iov_effect_names.is_empty() => {}
            _ => bail!("omega_iov metadata dimension mismatch"),
        }

        let mut coordinates = Vec::new();
        let mut population = vec![None; parameter_names.len()];
        for (parameter_index, (name, estimated)) in
            parameter_names.iter().zip(estimated_parameters).enumerate()
        {
            if *estimated {
                population[parameter_index] = Some(push_coordinate(
                    &mut coordinates,
                    format!("phi:{name}"),
                    InformationCoordinateKind::Population { parameter_index },
                ));
            }
        }
        let covariate_effects = covariate_effect_names
            .iter()
            .zip(covariate_estimated)
            .enumerate()
            .map(|(effect_index, (name, estimated))| {
                estimated.then(|| {
                    push_coordinate(
                        &mut coordinates,
                        name.clone(),
                        InformationCoordinateKind::CovariateEffect { effect_index },
                    )
                })
            })
            .collect();
        let omega = covariance_coordinates(
            &mut coordinates,
            random_effect_names,
            omega_structural,
            omega_estimated,
            false,
        );
        let omega_iov = match (omega_iov_structural, omega_iov_estimated) {
            (Some(structural), Some(estimated)) => covariance_coordinates(
                &mut coordinates,
                iov_effect_names,
                structural,
                estimated,
                true,
            ),
            _ => Vec::new(),
        };

        let mut residual = vec![ResidualCoordinates::default(); error_models.len()];
        for (output_index, residual_coordinates) in residual.iter_mut().enumerate() {
            let Some(model) = error_models.get(output_index) else {
                continue;
            };
            let output = error_models
                .output_name(output_index)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("output_{output_index}"));
            match *model {
                ResidualErrorModel::Constant { .. } | ResidualErrorModel::Exponential { .. } => {
                    if error_models.is_estimated(output_index) {
                        residual_coordinates.additive = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:sigma"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "sigma".to_string(),
                            },
                        ));
                    }
                }
                ResidualErrorModel::Proportional { .. } => {
                    if error_models.is_estimated(output_index) {
                        residual_coordinates.proportional = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:proportional"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "proportional".to_string(),
                            },
                        ));
                    }
                }
                ResidualErrorModel::Combined { .. } => {
                    let estimated = error_models.combined_component_estimated(output_index);
                    if estimated[0] {
                        residual_coordinates.additive = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:additive"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "additive".to_string(),
                            },
                        ));
                    }
                    if estimated[1] {
                        residual_coordinates.proportional = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:proportional"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "proportional".to_string(),
                            },
                        ));
                    }
                }
                ResidualErrorModel::CorrelatedCombined { .. } => {
                    let estimated =
                        error_models.correlated_combined_component_estimated(output_index);
                    if estimated[0] {
                        residual_coordinates.additive = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:additive"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "additive".to_string(),
                            },
                        ));
                    }
                    if estimated[1] {
                        residual_coordinates.proportional = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:proportional"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "proportional".to_string(),
                            },
                        ));
                    }
                    if estimated[2] {
                        residual_coordinates.correlation = Some(push_coordinate(
                            &mut coordinates,
                            format!("residual:{output}:correlation"),
                            InformationCoordinateKind::Residual {
                                output_index,
                                component: "correlation".to_string(),
                            },
                        ));
                    }
                }
            }
        }
        Ok(Self {
            coordinates,
            population,
            covariate_effects,
            omega,
            omega_iov,
            residual,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub(crate) fn residual(&self, output: usize) -> ResidualCoordinates {
        self.residual.get(output).copied().unwrap_or_default()
    }
}

fn validate_masks(
    width: usize,
    structural: &Array2<bool>,
    estimated: &Array2<bool>,
    label: &str,
) -> Result<()> {
    if structural.dim() != (width, width) || estimated.dim() != (width, width) {
        bail!("{label} mask dimension mismatch");
    }
    for row in 0..width {
        for column in 0..width {
            if structural[[row, column]] != structural[[column, row]]
                || estimated[[row, column]] != estimated[[column, row]]
                || (estimated[[row, column]] && !structural[[row, column]])
            {
                bail!("{label} masks must be symmetric and estimated entries structural");
            }
        }
    }
    Ok(())
}

fn push_coordinate(
    coordinates: &mut Vec<InformationCoordinate>,
    name: String,
    kind: InformationCoordinateKind,
) -> usize {
    let index = coordinates.len();
    coordinates.push(InformationCoordinate { index, name, kind });
    index
}

fn covariance_coordinates(
    coordinates: &mut Vec<InformationCoordinate>,
    names: &[String],
    structural: &Array2<bool>,
    estimated: &Array2<bool>,
    iov: bool,
) -> Vec<CovarianceCoordinate> {
    let mut result = Vec::new();
    for row in 0..names.len() {
        for column in 0..=row {
            if !structural[[row, column]] || !estimated[[row, column]] {
                continue;
            }
            let kind = if iov {
                InformationCoordinateKind::OmegaIov { row, column }
            } else {
                InformationCoordinateKind::Omega { row, column }
            };
            let prefix = if iov { "omega_iov" } else { "omega" };
            result.push(CovarianceCoordinate {
                coordinate: push_coordinate(
                    coordinates,
                    format!("{prefix}:{}:{}", names[row], names[column]),
                    kind,
                ),
                row,
                column,
            });
        }
    }
    result
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompleteDerivative {
    pub(crate) score: Vec<f64>,
    pub(crate) hessian: Array2<f64>,
}

impl CompleteDerivative {
    pub(crate) fn zero(width: usize) -> Self {
        Self {
            score: vec![0.0; width],
            hessian: Array2::zeros((width, width)),
        }
    }

    /// Add one Gaussian log-density contribution. `deviation` is x-mu.
    pub(crate) fn add_gaussian(
        &mut self,
        deviation: &[f64],
        covariance: &Array2<f64>,
        mean_coordinates: &[Option<usize>],
        covariance_coordinates: &[CovarianceCoordinate],
    ) -> Result<()> {
        let n = deviation.len();
        if covariance.dim() != (n, n) || mean_coordinates.len() != n {
            bail!("Gaussian derivative dimension mismatch");
        }
        if n == 0 {
            if covariance_coordinates.is_empty() {
                return self.ensure_finite();
            }
            bail!("zero-dimensional Gaussian prior has covariance coordinates");
        }
        let inverse = inverse_spd(covariance)?;
        let z = mat_vec(&inverse, deviation);
        if !z.iter().all(|value| value.is_finite()) {
            bail!("non-finite Gaussian derivative");
        }
        for (row, coordinate) in mean_coordinates.iter().enumerate() {
            let Some(coordinate) = coordinate else {
                continue;
            };
            self.score[*coordinate] += z[row];
            for (column, other) in mean_coordinates.iter().enumerate() {
                if let Some(other) = other {
                    self.hessian[[*coordinate, *other]] -= inverse[[row, column]];
                }
            }
        }
        for covariance_coordinate in covariance_coordinates {
            let basis = symmetric_basis(n, covariance_coordinate.row, covariance_coordinate.column);
            let ab = inverse.dot(&basis);
            let abz = ab.dot(&Array2::from_shape_vec((n, 1), z.clone())?);
            let quadratic = deviation
                .iter()
                .enumerate()
                .map(|(index, value)| value * abz[[index, 0]])
                .sum::<f64>();
            let coordinate = covariance_coordinate.coordinate;
            self.score[coordinate] += -0.5 * trace(&ab) + 0.5 * quadratic;

            for (mean_row, mean_coordinate) in mean_coordinates.iter().enumerate() {
                if let Some(mean_coordinate) = mean_coordinate {
                    let cross = -ab
                        .row(mean_row)
                        .iter()
                        .zip(&z)
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                    self.hessian[[*mean_coordinate, coordinate]] += cross;
                    self.hessian[[coordinate, *mean_coordinate]] += cross;
                }
            }
            for other in covariance_coordinates {
                let other_basis = symmetric_basis(n, other.row, other.column);
                let abs = inverse.dot(&other_basis);
                let trace_term = 0.5 * trace(&ab.dot(&abs));
                let first = ab.dot(&abs).dot(&inverse);
                let second = abs.dot(&ab).dot(&inverse);
                let quadratic_term = -0.5 * quadratic_form(deviation, &(first + second));
                self.hessian[[coordinate, other.coordinate]] += trace_term + quadratic_term;
            }
        }
        self.ensure_finite()
    }

    pub(crate) fn add_population_prior(
        &mut self,
        eta: &[f64],
        covariance: &Array2<f64>,
        random_effect_parameter_indices: &[usize],
        layout: &InformationLayout,
    ) -> Result<()> {
        if eta.len() != random_effect_parameter_indices.len() {
            bail!("eta dimension mismatch");
        }
        let means = random_effect_parameter_indices
            .iter()
            .map(|index| layout.population.get(*index).copied().flatten())
            .collect::<Vec<_>>();
        self.add_gaussian(eta, covariance, &means, &layout.omega)
    }

    pub(crate) fn add_covariate_population_prior(
        &mut self,
        eta: &[f64],
        covariance: &Array2<f64>,
        random_effect_parameter_indices: &[usize],
        effect_parameter_indices: &[usize],
        subject_design_values: &[f64],
        layout: &InformationLayout,
    ) -> Result<()> {
        if eta.len() != random_effect_parameter_indices.len()
            || effect_parameter_indices.len() != subject_design_values.len()
            || effect_parameter_indices.len() != layout.covariate_effects.len()
        {
            bail!("covariate Gaussian derivative dimension mismatch");
        }
        if eta.is_empty() {
            return Ok(());
        }
        let mut columns = Vec::<Vec<f64>>::new();
        let mut coordinates = Vec::new();
        for (row, parameter_index) in random_effect_parameter_indices.iter().copied().enumerate() {
            if let Some(coordinate) = layout.population.get(parameter_index).copied().flatten() {
                let mut column = vec![0.0; eta.len()];
                column[row] = 1.0;
                columns.push(column);
                coordinates.push(coordinate);
            }
        }
        for (effect_index, coordinate) in layout.covariate_effects.iter().enumerate() {
            let Some(coordinate) = coordinate else {
                continue;
            };
            let Some(row) = random_effect_parameter_indices
                .iter()
                .position(|parameter| *parameter == effect_parameter_indices[effect_index])
            else {
                bail!("estimated covariate effect does not target an IIV coordinate");
            };
            let mut column = vec![0.0; eta.len()];
            column[row] = subject_design_values[effect_index];
            columns.push(column);
            coordinates.push(*coordinate);
        }
        let design = Array2::from_shape_fn((eta.len(), columns.len()), |(row, column)| {
            columns[column][row]
        });
        self.add_design_mean_prior(eta, covariance, &design, &coordinates, &layout.omega)
    }

    pub(crate) fn add_iov_prior(
        &mut self,
        kappa: &[f64],
        covariance: &Array2<f64>,
        layout: &InformationLayout,
    ) -> Result<()> {
        self.add_gaussian(
            kappa,
            covariance,
            &vec![None; kappa.len()],
            &layout.omega_iov,
        )
    }

    /// Add one generalized subject-design Gaussian log-density contribution.
    ///
    /// `design` is an `n_random_effects × n_coefficients` matrix where each
    /// column is the design vector for the corresponding coefficient coordinate.
    /// Score: `score_c = A_col_c' * W * eta`
    /// Hessian: `H_cc = -A' * W * A` and `H_{c,Omega_h} = -A_col_c' * W * S_h * W * eta`
    pub(crate) fn add_design_mean_prior(
        &mut self,
        eta: &[f64],
        covariance: &Array2<f64>,
        design: &Array2<f64>,
        coefficient_coordinates: &[usize],
        covariance_coordinates: &[CovarianceCoordinate],
    ) -> Result<()> {
        let n_random = eta.len();
        let n_coefficients = coefficient_coordinates.len();
        if covariance.dim() != (n_random, n_random) || design.dim() != (n_random, n_coefficients) {
            bail!("design mean-prior derivative dimension mismatch");
        }
        let inverse = inverse_spd(covariance)?;
        let z = mat_vec(&inverse, eta);
        if !z.iter().all(|value| value.is_finite()) {
            bail!("non-finite design mean-prior derivative");
        }
        // Score: score_c = A[:,c]' * z
        for (col, coordinate) in coefficient_coordinates.iter().enumerate() {
            let score_c: f64 = design
                .column(col)
                .iter()
                .zip(&z)
                .map(|(a, zi)| a * zi)
                .sum();
            self.score[*coordinate] += score_c;
        }
        // Hessian beta-beta: H_c1,c2 = -A[:,c1]' * W * A[:,c2]
        for (c1, coord1) in coefficient_coordinates.iter().enumerate() {
            for (c2, coord2) in coefficient_coordinates.iter().enumerate() {
                let mut value = 0.0;
                for row in 0..n_random {
                    let wa_c2 = (0..n_random)
                        .map(|k| inverse[[row, k]] * design[[k, c2]])
                        .sum::<f64>();
                    value += design[[row, c1]] * wa_c2;
                }
                self.hessian[[*coord1, *coord2]] -= value;
            }
        }
        // Hessian beta-Omega and Omega score: same as add_gaussian for Omega-on-Omega
        for covariance_coordinate in covariance_coordinates {
            let basis = symmetric_basis(
                n_random,
                covariance_coordinate.row,
                covariance_coordinate.column,
            );
            let ab = inverse.dot(&basis);
            let abz = ab.dot(&Array2::from_shape_vec((n_random, 1), z.clone())?);
            let quadratic = eta
                .iter()
                .enumerate()
                .map(|(index, value)| value * abz[[index, 0]])
                .sum::<f64>();
            let omega_coord = covariance_coordinate.coordinate;
            self.score[omega_coord] += -0.5 * trace(&ab) + 0.5 * quadratic;
            // Beta-Omega cross: H_{c,Omega_h} = -A[:,c]' * W * S_h * z
            for (col, coordinate) in coefficient_coordinates.iter().enumerate() {
                let cross = -(0..n_random)
                    .map(|row| {
                        design[[row, col]] * (0..n_random).map(|k| ab[[row, k]] * z[k]).sum::<f64>()
                    })
                    .sum::<f64>();
                self.hessian[[*coordinate, omega_coord]] += cross;
                self.hessian[[omega_coord, *coordinate]] += cross;
            }
            // Omega-Omega (same as original add_gaussian)
            for other in covariance_coordinates {
                let other_basis = symmetric_basis(n_random, other.row, other.column);
                let abs = inverse.dot(&other_basis);
                let trace_term = 0.5 * trace(&ab.dot(&abs));
                let first = ab.dot(&abs).dot(&inverse);
                let second = abs.dot(&ab).dot(&inverse);
                let quadratic_term = -0.5 * quadratic_form(eta, &(first + second));
                self.hessian[[omega_coord, other.coordinate]] += trace_term + quadratic_term;
            }
        }
        self.ensure_finite()
    }

    pub(crate) fn add_predictions<P: Predictions>(
        &mut self,
        predictions: &P,
        error_models: &ParametricErrorModels,
        layout: &InformationLayout,
    ) -> Result<()> {
        let mut failure = None;
        predictions.for_each_prediction(|prediction: &Prediction| {
            if failure.is_some() {
                return;
            }
            let Some(model) = error_models.get(prediction.outeq()).copied() else {
                return;
            };
            if let Err(error) = self.add_residual(
                prediction.outeq(),
                prediction.observation(),
                prediction.prediction(),
                prediction.censoring(),
                model,
                layout,
            ) {
                failure = Some(error);
            }
        });
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Strict retained-Markov score semantics: censoring and every active or
    /// equal residual likelihood-floor branch are unsupported.
    pub(crate) fn add_predictions_strict<P: Predictions>(
        &mut self,
        predictions: &P,
        error_models: &ParametricErrorModels,
        layout: &InformationLayout,
    ) -> Result<()> {
        let mut failure = None;
        predictions.for_each_prediction(|prediction: &Prediction| {
            if failure.is_some() || prediction.observation().is_none() {
                return;
            }
            let Some(model) = error_models.get(prediction.outeq()).copied() else {
                return;
            };
            let raw_scale = match model {
                ResidualErrorModel::Constant { a } => a,
                ResidualErrorModel::Proportional { b } => b * prediction.prediction().abs(),
                ResidualErrorModel::Combined { a, b } => {
                    (a * a + b * b * prediction.prediction().powi(2)).sqrt()
                }
                ResidualErrorModel::CorrelatedCombined { a, b, rho } => (a * a
                    + 2.0 * rho * a * b * prediction.prediction()
                    + b * b * prediction.prediction().powi(2))
                .sqrt(),
                ResidualErrorModel::Exponential { sigma } => sigma,
            };
            if prediction.censoring() != Censor::None {
                failure = Some(anyhow::anyhow!(
                    "retained Markov scores are unsupported for censored observations"
                ));
            } else if !raw_scale.is_finite() || raw_scale <= f64::EPSILON.sqrt() {
                failure = Some(anyhow::anyhow!(
                    "retained Markov scores are unsupported on an active or equal likelihood-floor branch"
                ));
            } else if let Err(error) = self.add_residual(
                prediction.outeq(),
                prediction.observation(),
                prediction.prediction(),
                prediction.censoring(),
                model,
                layout,
            ) {
                failure = Some(error);
            }
        });
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(crate) fn add_residual(
        &mut self,
        output: usize,
        observation: Option<f64>,
        prediction: f64,
        censor: Censor,
        model: ResidualErrorModel,
        layout: &InformationLayout,
    ) -> Result<()> {
        let Some(observation) = observation else {
            return Ok(());
        };
        if censor != Censor::None {
            bail!("analytic information is unsupported for censored observations");
        }
        if !observation.is_finite() || !prediction.is_finite() {
            bail!("non-finite residual derivative input");
        }
        let coordinates = layout.residual(output);
        match model {
            ResidualErrorModel::Constant { a } => {
                if scale_floor_branch(a, "constant")? == ScaleFloorBranch::Above {
                    add_simple_residual(self, coordinates.additive, observation - prediction, a)?;
                }
            }
            ResidualErrorModel::Proportional { b } => {
                let raw_sigma = b * prediction.abs();
                if scale_floor_branch(raw_sigma, "proportional")? == ScaleFloorBranch::Above {
                    let residual = observation - prediction;
                    if let Some(coordinate) = coordinates.proportional {
                        let q = prediction * prediction;
                        add_scale_derivative(self, coordinate, residual, b, q)?;
                    }
                }
            }
            ResidualErrorModel::Combined { a, b } => {
                let raw_sigma = (a * a + b * b * prediction * prediction).sqrt();
                if scale_floor_branch(raw_sigma, "combined")? == ScaleFloorBranch::Above {
                    add_combined_residual(
                        self,
                        coordinates,
                        observation - prediction,
                        prediction,
                        a,
                        b,
                    )?;
                }
            }
            ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                let raw_sigma =
                    (a * a + 2.0 * rho * a * b * prediction + b * b * prediction * prediction)
                        .sqrt();
                if scale_floor_branch(raw_sigma, "correlated-combined")? == ScaleFloorBranch::Above
                {
                    add_correlated_combined_residual(
                        self,
                        coordinates,
                        observation - prediction,
                        prediction,
                        a,
                        b,
                        rho,
                    )?;
                }
            }
            ResidualErrorModel::Exponential { sigma } => {
                if observation <= 0.0 || prediction <= 0.0 {
                    bail!("exponential residual information requires positive observation and prediction");
                }
                if scale_floor_branch(sigma, "exponential")? == ScaleFloorBranch::Above {
                    add_simple_residual(
                        self,
                        coordinates.additive,
                        observation.ln() - prediction.ln(),
                        sigma,
                    )?;
                }
            }
        }
        self.ensure_finite()
    }

    fn ensure_finite(&self) -> Result<()> {
        if !self.score.iter().all(|value| value.is_finite())
            || !self.hessian.iter().all(|value| value.is_finite())
        {
            bail!("non-finite complete-data derivative");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScaleFloorBranch {
    Below,
    Above,
}

fn scale_floor_branch(raw_sigma: f64, family: &str) -> Result<ScaleFloorBranch> {
    if !raw_sigma.is_finite() || raw_sigma < 0.0 {
        bail!("{family} residual scale must be finite and nonnegative");
    }
    let floor = f64::EPSILON.sqrt();
    if raw_sigma == floor {
        bail!(
            "{family} residual scale is exactly at the nondifferentiable likelihood floor boundary"
        );
    }
    Ok(if raw_sigma < floor {
        ScaleFloorBranch::Below
    } else {
        ScaleFloorBranch::Above
    })
}

fn add_simple_residual(
    derivative: &mut CompleteDerivative,
    coordinate: Option<usize>,
    residual: f64,
    sigma: f64,
) -> Result<()> {
    let Some(coordinate) = coordinate else {
        return Ok(());
    };
    if !sigma.is_finite() || sigma <= 0.0 {
        bail!("residual standard deviation must be finite and positive");
    }
    add_scale_derivative(derivative, coordinate, residual, sigma, 1.0)
}

fn add_scale_derivative(
    derivative: &mut CompleteDerivative,
    coordinate: usize,
    residual: f64,
    sigma_parameter: f64,
    prediction_squared: f64,
) -> Result<()> {
    if sigma_parameter <= 0.0 || prediction_squared <= 0.0 {
        bail!("invalid residual derivative scale");
    }
    let residual_squared = residual * residual;
    derivative.score[coordinate] +=
        -1.0 / sigma_parameter + residual_squared / sigma_parameter.powi(3) / prediction_squared;
    derivative.hessian[[coordinate, coordinate]] += 1.0 / sigma_parameter.powi(2)
        - 3.0 * residual_squared / sigma_parameter.powi(4) / prediction_squared;
    Ok(())
}

fn add_combined_residual(
    derivative: &mut CompleteDerivative,
    coordinates: ResidualCoordinates,
    residual: f64,
    prediction: f64,
    a: f64,
    b: f64,
) -> Result<()> {
    if a < 0.0 || b < 0.0 || !a.is_finite() || !b.is_finite() {
        bail!("combined residual components must be finite and nonnegative");
    }
    let f2 = prediction * prediction;
    let variance = a * a + b * b * f2;
    let r2 = residual * residual;
    let common = 1.0 / variance - r2 / variance.powi(2);
    let curvature = -1.0 / variance.powi(2) + 2.0 * r2 / variance.powi(3);
    if let Some(ai) = coordinates.additive {
        derivative.score[ai] -= a * common;
        derivative.hessian[[ai, ai]] -= common + 2.0 * a * a * curvature;
    }
    if let Some(bi) = coordinates.proportional {
        derivative.score[bi] -= b * f2 * common;
        derivative.hessian[[bi, bi]] -= f2 * common + 2.0 * b * b * f2 * f2 * curvature;
    }
    if let (Some(ai), Some(bi)) = (coordinates.additive, coordinates.proportional) {
        let cross = -2.0 * a * b * f2 * curvature;
        derivative.hessian[[ai, bi]] += cross;
        derivative.hessian[[bi, ai]] += cross;
    }
    Ok(())
}

fn add_correlated_combined_residual(
    derivative: &mut CompleteDerivative,
    coordinates: ResidualCoordinates,
    residual: f64,
    prediction: f64,
    a: f64,
    b: f64,
    rho: f64,
) -> Result<()> {
    if !a.is_finite()
        || a <= 0.0
        || !b.is_finite()
        || b <= 0.0
        || !rho.is_finite()
        || rho <= -1.0
        || rho >= 1.0
    {
        bail!("correlated-combined residual components are outside their declared domains");
    }
    let f = prediction;
    let f2 = f * f;
    let variance = a * a + 2.0 * rho * a * b * f + b * b * f2;
    if !variance.is_finite() || variance <= 0.0 {
        bail!("correlated-combined residual variance must be finite and positive");
    }
    let residual_squared = residual * residual;
    let common = 1.0 / variance - residual_squared / variance.powi(2);
    let curvature = -1.0 / variance.powi(2) + 2.0 * residual_squared / variance.powi(3);
    let indices = [
        coordinates.additive,
        coordinates.proportional,
        coordinates.correlation,
    ];
    let first = [
        2.0 * (a + rho * b * f),
        2.0 * (b * f2 + rho * a * f),
        2.0 * a * b * f,
    ];
    let second = [
        [2.0, 2.0 * rho * f, 2.0 * b * f],
        [2.0 * rho * f, 2.0 * f2, 2.0 * a * f],
        [2.0 * b * f, 2.0 * a * f, 0.0],
    ];
    for left in 0..3 {
        let Some(left_index) = indices[left] else {
            continue;
        };
        derivative.score[left_index] -= 0.5 * common * first[left];
        for right in 0..3 {
            let Some(right_index) = indices[right] else {
                continue;
            };
            derivative.hessian[[left_index, right_index]] -=
                0.5 * (common * second[left][right] + curvature * first[left] * first[right]);
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct InformationRecursion {
    layout: InformationLayout,
    cycles: usize,
    delta: Vec<f64>,
    g: Array2<f64>,
    complete_hessian: Array2<f64>,
    failure: Option<InformationStatus>,
}

impl InformationRecursion {
    pub(crate) fn new(layout: InformationLayout) -> Self {
        let width = layout.len();
        Self {
            layout,
            cycles: 0,
            delta: vec![0.0; width],
            g: Array2::zeros((width, width)),
            complete_hessian: Array2::zeros((width, width)),
            failure: None,
        }
    }

    pub(crate) fn layout(&self) -> &InformationLayout {
        &self.layout
    }

    /// Apply one SA update from full-dataset chain replicates.
    pub(crate) fn update(&mut self, replicates: &[CompleteDerivative], gamma: f64) {
        if gamma == 0.0 || self.failure.is_some() {
            return;
        }
        let width = self.layout.len();
        if replicates.is_empty()
            || !gamma.is_finite()
            || gamma <= 0.0
            || gamma > 1.0
            || replicates.iter().any(|replicate| {
                replicate.score.len() != width
                    || replicate.hessian.dim() != (width, width)
                    || !replicate.score.iter().all(|value| value.is_finite())
                    || !replicate.hessian.iter().all(|value| value.is_finite())
            })
        {
            self.failure = Some(InformationStatus::NonFinite);
            return;
        }
        let count = replicates.len() as f64;
        let mut mean_score = vec![0.0; width];
        let mut mean_hessian = Array2::<f64>::zeros((width, width));
        let mut mean_augmented = Array2::<f64>::zeros((width, width));
        for replicate in replicates {
            for row in 0..width {
                mean_score[row] += replicate.score[row] / count;
                for column in 0..width {
                    mean_hessian[[row, column]] += replicate.hessian[[row, column]] / count;
                    mean_augmented[[row, column]] += (replicate.hessian[[row, column]]
                        + replicate.score[row] * replicate.score[column])
                        / count;
                }
            }
        }
        for row in 0..width {
            self.delta[row] += gamma * (mean_score[row] - self.delta[row]);
            for column in 0..width {
                self.complete_hessian[[row, column]] +=
                    gamma * (mean_hessian[[row, column]] - self.complete_hessian[[row, column]]);
                self.g[[row, column]] +=
                    gamma * (mean_augmented[[row, column]] - self.g[[row, column]]);
            }
        }
        self.cycles += 1;
    }

    pub(crate) fn mark_unavailable(&mut self, status: InformationStatus) {
        if self.failure.is_none() {
            self.failure = Some(status);
        }
    }

    pub(crate) fn diagnostics(&self) -> InformationDiagnostics {
        let width = self.layout.len();
        let mut observed_hessian = self.g.clone();
        for row in 0..width {
            for column in 0..width {
                observed_hessian[[row, column]] -= self.delta[row] * self.delta[column];
            }
        }
        let symmetric = is_finite_symmetric(&observed_hessian)
            && is_finite_symmetric(&self.g)
            && is_finite_symmetric(&self.complete_hessian);
        if symmetric {
            symmetrize_roundoff(&mut observed_hessian);
        }
        let observed_information = observed_hessian.mapv(|value| -value);
        let status = if let Some(status) = &self.failure {
            status.clone()
        } else if width == 0 {
            InformationStatus::NoFreeCoordinates
        } else if !symmetric {
            InformationStatus::NonFinite
        } else if cholesky_lower(&observed_information).is_err() {
            InformationStatus::ObservedInformationNotPositiveDefinite
        } else {
            InformationStatus::Available
        };
        InformationDiagnostics {
            coordinates: self.layout.coordinates.clone(),
            recursion_cycles: self.cycles,
            delta: self.delta.clone(),
            g: rows(&self.g),
            expected_complete_hessian: rows(&self.complete_hessian),
            observed_hessian: rows(&observed_hessian),
            observed_information: rows(&observed_information),
            status,
        }
    }
}

/// Derive free-coordinate population uncertainty from observed-information diagnostics.
///
/// Inverts only when [`InformationStatus::Available`]; never applies regularization,
/// repair, or fallback. The returned covariance and standard errors are in the
/// estimation (φ) space coordinated with the diagnostic free-coordinate order.
pub(crate) fn derive_population_uncertainty(
    diagnostics: &InformationDiagnostics,
) -> PopulationUncertaintyDiagnostics {
    if diagnostics.status != InformationStatus::Available {
        let reason = match &diagnostics.status {
            InformationStatus::NonFinite => PopulationUncertaintyUnavailableReason::NonFinite,
            InformationStatus::ObservedInformationNotPositiveDefinite => {
                PopulationUncertaintyUnavailableReason::ObservedInformationNotPositiveDefinite
            }
            other => PopulationUncertaintyUnavailableReason::SourceUnavailable(other.clone()),
        };
        return unavailable_population_uncertainty(diagnostics, reason);
    }

    let n = diagnostics.coordinates.len();
    if n == 0 {
        return unavailable_population_uncertainty(
            diagnostics,
            PopulationUncertaintyUnavailableReason::SourceUnavailable(
                InformationStatus::NoFreeCoordinates,
            ),
        );
    }

    let observed_information = match rows_to_array2(&diagnostics.observed_information, n) {
        Ok(matrix) => matrix,
        Err(_) => {
            return unavailable_population_uncertainty(
                diagnostics,
                PopulationUncertaintyUnavailableReason::InversionFailed,
            );
        }
    };
    if !observed_information.iter().all(|value| value.is_finite()) {
        return unavailable_population_uncertainty(
            diagnostics,
            PopulationUncertaintyUnavailableReason::NonFinite,
        );
    }

    // Strict SPD classification and inversion through one unmodified Cholesky factor.
    let lower = match cholesky_lower(&observed_information) {
        Ok(lower) => lower,
        Err(_) => {
            return unavailable_population_uncertainty(
                diagnostics,
                PopulationUncertaintyUnavailableReason::ObservedInformationNotPositiveDefinite,
            );
        }
    };
    let free_covariance = match inverse_spd_from_cholesky(&lower) {
        Ok(inverse) => inverse,
        Err(_) => {
            return unavailable_population_uncertainty(
                diagnostics,
                PopulationUncertaintyUnavailableReason::InversionFailed,
            );
        }
    };

    let (minimum_eigenvalue, maximum_eigenvalue) =
        match eigenvalue_extrema_symmetric(&observed_information) {
            Ok(extrema) => extrema,
            Err(_) => {
                return unavailable_population_uncertainty(
                    diagnostics,
                    PopulationUncertaintyUnavailableReason::InversionFailed,
                );
            }
        };
    if !minimum_eigenvalue.is_finite() || !maximum_eigenvalue.is_finite() {
        return unavailable_population_uncertainty(
            diagnostics,
            PopulationUncertaintyUnavailableReason::NonFinite,
        );
    }
    if minimum_eigenvalue <= 0.0 {
        return unavailable_population_uncertainty(
            diagnostics,
            PopulationUncertaintyUnavailableReason::ObservedInformationNotPositiveDefinite,
        );
    }
    let spectral_condition_number = maximum_eigenvalue / minimum_eigenvalue;
    if !spectral_condition_number.is_finite() {
        return unavailable_population_uncertainty(
            diagnostics,
            PopulationUncertaintyUnavailableReason::NonFinite,
        );
    }

    PopulationUncertaintyDiagnostics {
        coordinates: diagnostics.coordinates.clone(),
        free_standard_errors: Some(
            (0..n)
                .map(|index| free_covariance[[index, index]].sqrt())
                .collect(),
        ),
        free_covariance: Some(rows(&free_covariance)),
        spectral_condition_number: Some(spectral_condition_number),
        status: PopulationUncertaintyStatus::Available,
        regularization: PopulationUncertaintyRegularization::None,
    }
}

fn unavailable_population_uncertainty(
    diagnostics: &InformationDiagnostics,
    reason: PopulationUncertaintyUnavailableReason,
) -> PopulationUncertaintyDiagnostics {
    let mut result = PopulationUncertaintyDiagnostics::unavailable(reason);
    result.coordinates = diagnostics.coordinates.clone();
    result
}

fn rows_to_array2(rows: &[Vec<f64>], n: usize) -> Result<Array2<f64>> {
    if rows.len() != n || rows.iter().any(|row| row.len() != n) {
        bail!("observed-information matrix dimensions must match coordinate count");
    }
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    Ok(Array2::from_shape_vec((n, n), flat)?)
}

fn inverse_spd(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let lower = cholesky_lower(matrix)?;
    inverse_spd_from_cholesky(&lower)
}

fn symmetric_basis(width: usize, row: usize, column: usize) -> Array2<f64> {
    let mut basis = Array2::zeros((width, width));
    basis[[row, column]] = 1.0;
    basis[[column, row]] = 1.0;
    basis
}

fn mat_vec(matrix: &Array2<f64>, vector: &[f64]) -> Vec<f64> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[[row, column]] * vector[column])
                .sum()
        })
        .collect()
}

fn quadratic_form(vector: &[f64], matrix: &Array2<f64>) -> f64 {
    let product = mat_vec(matrix, vector);
    vector
        .iter()
        .zip(product)
        .map(|(left, right)| left * right)
        .sum()
}

fn trace(matrix: &Array2<f64>) -> f64 {
    (0..matrix.nrows())
        .map(|index| matrix[[index, index]])
        .sum()
}

fn rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn is_finite_symmetric(matrix: &Array2<f64>) -> bool {
    if matrix.nrows() != matrix.ncols() || !matrix.iter().all(|value| value.is_finite()) {
        return false;
    }
    for row in 0..matrix.nrows() {
        for column in 0..row {
            let scale = matrix[[row, column]]
                .abs()
                .max(matrix[[column, row]].abs())
                .max(1.0);
            if (matrix[[row, column]] - matrix[[column, row]]).abs() > 64.0 * f64::EPSILON * scale {
                return false;
            }
        }
    }
    true
}

fn symmetrize_roundoff(matrix: &mut Array2<f64>) {
    for row in 0..matrix.nrows() {
        for column in 0..row {
            let value = 0.5 * matrix[[row, column]] + 0.5 * matrix[[column, row]];
            matrix[[row, column]] = value;
            matrix[[column, row]] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_coordinate_layout() -> InformationLayout {
        InformationLayout {
            coordinates: vec![InformationCoordinate {
                index: 0,
                name: "phi:x".into(),
                kind: InformationCoordinateKind::Population { parameter_index: 0 },
            }],
            population: vec![Some(0)],
            covariate_effects: Vec::new(),
            omega: Vec::new(),
            omega_iov: Vec::new(),
            residual: Vec::new(),
        }
    }

    fn two_coordinate_layout() -> InformationLayout {
        let mut layout = one_coordinate_layout();
        layout.coordinates.push(InformationCoordinate {
            index: 1,
            name: "phi:y".into(),
            kind: InformationCoordinateKind::Population { parameter_index: 1 },
        });
        layout.population.push(Some(1));
        layout
    }

    fn diagnostics_with_information(
        coordinates: Vec<InformationCoordinate>,
        observed_information: Vec<Vec<f64>>,
        status: InformationStatus,
    ) -> InformationDiagnostics {
        let width = coordinates.len();
        InformationDiagnostics {
            coordinates,
            recursion_cycles: 1,
            delta: vec![0.0; width],
            g: vec![vec![0.0; width]; width],
            expected_complete_hessian: vec![vec![0.0; width]; width],
            observed_hessian: observed_information
                .iter()
                .map(|row| row.iter().map(|value| -*value).collect())
                .collect(),
            observed_information,
            status,
        }
    }

    #[test]
    fn population_uncertainty_inverts_one_and_two_coordinate_information() {
        let one = diagnostics_with_information(
            one_coordinate_layout().coordinates,
            vec![vec![4.0]],
            InformationStatus::Available,
        );
        let one_uncertainty = derive_population_uncertainty(&one);
        assert_eq!(
            one_uncertainty.status,
            PopulationUncertaintyStatus::Available
        );
        assert_eq!(one_uncertainty.free_covariance, Some(vec![vec![0.25]]));
        assert_eq!(one_uncertainty.free_standard_errors, Some(vec![0.5]));
        assert_eq!(one_uncertainty.spectral_condition_number, Some(1.0));
        assert_eq!(
            one_uncertainty.regularization,
            PopulationUncertaintyRegularization::None
        );

        let two = diagnostics_with_information(
            two_coordinate_layout().coordinates,
            vec![vec![4.0, 2.0], vec![2.0, 3.0]],
            InformationStatus::Available,
        );
        let two_uncertainty = derive_population_uncertainty(&two);
        let covariance = two_uncertainty.free_covariance.unwrap();
        assert!((covariance[0][0] - 0.375).abs() < 1e-12);
        assert!((covariance[0][1] + 0.25).abs() < 1e-12);
        assert!((covariance[1][0] + 0.25).abs() < 1e-12);
        assert!((covariance[1][1] - 0.5).abs() < 1e-12);
        let standard_errors = two_uncertainty.free_standard_errors.unwrap();
        assert!((standard_errors[0] - 0.375_f64.sqrt()).abs() < 1e-12);
        assert!((standard_errors[1] - 0.5_f64.sqrt()).abs() < 1e-12);
        let expected_condition = (7.0 + 17.0_f64.sqrt()) / (7.0 - 17.0_f64.sqrt());
        assert!(
            (two_uncertainty.spectral_condition_number.unwrap() - expected_condition).abs() < 1e-12
        );
    }

    #[test]
    fn population_uncertainty_has_typed_unavailable_reasons() {
        let coordinate = one_coordinate_layout().coordinates;
        for (status, expected) in [
            (
                InformationStatus::Ineligible("not accumulated".into()),
                PopulationUncertaintyUnavailableReason::SourceUnavailable(
                    InformationStatus::Ineligible("not accumulated".into()),
                ),
            ),
            (
                InformationStatus::NonFinite,
                PopulationUncertaintyUnavailableReason::NonFinite,
            ),
            (
                InformationStatus::ObservedInformationNotPositiveDefinite,
                PopulationUncertaintyUnavailableReason::ObservedInformationNotPositiveDefinite,
            ),
        ] {
            let diagnostics =
                diagnostics_with_information(coordinate.clone(), vec![vec![1.0]], status);
            assert_eq!(
                derive_population_uncertainty(&diagnostics).status,
                PopulationUncertaintyStatus::Unavailable(expected)
            );
        }

        let nonfinite = diagnostics_with_information(
            coordinate.clone(),
            vec![vec![f64::NAN]],
            InformationStatus::Available,
        );
        assert_eq!(
            derive_population_uncertainty(&nonfinite).status,
            PopulationUncertaintyStatus::Unavailable(
                PopulationUncertaintyUnavailableReason::NonFinite
            )
        );

        let non_positive_definite = diagnostics_with_information(
            coordinate.clone(),
            vec![vec![-1.0]],
            InformationStatus::Available,
        );
        assert_eq!(
            derive_population_uncertainty(&non_positive_definite).status,
            PopulationUncertaintyStatus::Unavailable(
                PopulationUncertaintyUnavailableReason::ObservedInformationNotPositiveDefinite
            )
        );

        let inversion_overflow = diagnostics_with_information(
            coordinate,
            vec![vec![f64::from_bits(1)]],
            InformationStatus::Available,
        );
        assert_eq!(
            derive_population_uncertainty(&inversion_overflow).status,
            PopulationUncertaintyStatus::Unavailable(
                PopulationUncertaintyUnavailableReason::InversionFailed
            )
        );
    }

    #[test]
    fn one_dimensional_gaussian_mean_score_and_hessian_are_exact() {
        let layout = one_coordinate_layout();
        let mut derivative = CompleteDerivative::zero(1);
        derivative
            .add_gaussian(&[2.0], &ndarray::array![[4.0]], &[Some(0)], &[])
            .unwrap();
        assert_eq!(derivative.score, vec![0.5]);
        assert_eq!(derivative.hessian[[0, 0]], -0.25);
        assert_eq!(layout.len(), 1);
    }

    #[test]
    fn population_uncertainty_layout_preserves_mixed_fixed_free_masks_and_order() {
        use crate::estimation::ParametricErrorModel;

        let structural = ndarray::array![[true, true], [true, true]];
        let estimated = ndarray::array![[true, true], [true, false]];
        let iov_mask = ndarray::array![[true]];
        let models = ParametricErrorModels::new()
            .add(
                0,
                "fixed",
                ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
            )
            .add(
                1,
                "combined",
                ParametricErrorModel::new(ResidualErrorModel::combined(0.2, 0.3))
                    .fixed_combined_additive(),
            )
            .add(2, "prop", ResidualErrorModel::proportional(0.1).into());
        let layout = InformationLayout::new(
            &["a".into(), "b".into(), "c".into()],
            &[true, false, true],
            &[],
            &[],
            &["a".into(), "c".into()],
            &structural,
            &estimated,
            &["c".into()],
            Some(&iov_mask),
            Some(&iov_mask),
            &models,
        )
        .unwrap();
        assert_eq!(
            layout
                .coordinates
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "phi:a",
                "phi:c",
                "omega:a:a",
                "omega:c:a",
                "omega_iov:c:c",
                "residual:combined:proportional",
                "residual:prop:proportional",
            ]
        );
        let coordinates = layout.coordinates.clone();
        let width = coordinates.len();
        let observed_information = (0..width)
            .map(|row| {
                (0..width)
                    .map(|column| if row == column { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let diagnostics = diagnostics_with_information(
            coordinates.clone(),
            observed_information,
            InformationStatus::Available,
        );
        let uncertainty = derive_population_uncertainty(&diagnostics);
        assert_eq!(uncertainty.coordinates, coordinates);
        assert_eq!(uncertainty.status, PopulationUncertaintyStatus::Available);
        assert_eq!(uncertainty.free_standard_errors, Some(vec![1.0; width]));
    }

    #[test]
    fn two_dimensional_gaussian_has_exact_mean_covariance_cross_block() {
        let coordinates = vec![
            CovarianceCoordinate {
                coordinate: 2,
                row: 0,
                column: 0,
            },
            CovarianceCoordinate {
                coordinate: 3,
                row: 1,
                column: 0,
            },
            CovarianceCoordinate {
                coordinate: 4,
                row: 1,
                column: 1,
            },
        ];
        let mut derivative = CompleteDerivative::zero(5);
        derivative
            .add_gaussian(
                &[1.0, -2.0],
                &ndarray::array![[2.0, 1.0], [1.0, 2.0]],
                &[Some(0), Some(1)],
                &coordinates,
            )
            .unwrap();
        let expected_score = [4.0 / 3.0, -5.0 / 3.0, 5.0 / 9.0, -17.0 / 9.0, 19.0 / 18.0];
        let expected_hessian = ndarray::array![
            [-2.0 / 3.0, 1.0 / 3.0, -8.0 / 9.0, 14.0 / 9.0, -5.0 / 9.0],
            [1.0 / 3.0, -2.0 / 3.0, 4.0 / 9.0, -13.0 / 9.0, 10.0 / 9.0],
            [
                -8.0 / 9.0,
                4.0 / 9.0,
                -26.0 / 27.0,
                50.0 / 27.0,
                -37.0 / 54.0
            ],
            [
                14.0 / 9.0,
                -13.0 / 9.0,
                50.0 / 27.0,
                -107.0 / 27.0,
                59.0 / 27.0
            ],
            [
                -5.0 / 9.0,
                10.0 / 9.0,
                -37.0 / 54.0,
                59.0 / 27.0,
                -44.0 / 27.0
            ],
        ];
        for (actual, expected) in derivative.score.iter().zip(expected_score) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in derivative.hessian.iter().zip(expected_hessian.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        // The raw symmetric off-diagonal covariance coordinate has no hidden
        // half-vectorization factor: its score is exactly -17/9.
        assert!((derivative.score[3] + 17.0 / 9.0).abs() < 1e-12);
    }

    fn residual_layout() -> InformationLayout {
        InformationLayout {
            coordinates: vec![
                InformationCoordinate {
                    index: 0,
                    name: "a".into(),
                    kind: InformationCoordinateKind::Residual {
                        output_index: 0,
                        component: "additive".into(),
                    },
                },
                InformationCoordinate {
                    index: 1,
                    name: "b".into(),
                    kind: InformationCoordinateKind::Residual {
                        output_index: 0,
                        component: "proportional".into(),
                    },
                },
            ],
            population: Vec::new(),
            covariate_effects: Vec::new(),
            omega: Vec::new(),
            omega_iov: Vec::new(),
            residual: vec![ResidualCoordinates {
                additive: Some(0),
                proportional: Some(1),
                correlation: None,
            }],
        }
    }

    #[test]
    fn multiple_iov_occasions_aggregate_into_one_complete_derivative() {
        let layout = InformationLayout {
            coordinates: vec![InformationCoordinate {
                index: 0,
                name: "omega_iov:k:k".into(),
                kind: InformationCoordinateKind::OmegaIov { row: 0, column: 0 },
            }],
            population: Vec::new(),
            covariate_effects: Vec::new(),
            omega: Vec::new(),
            omega_iov: vec![CovarianceCoordinate {
                coordinate: 0,
                row: 0,
                column: 0,
            }],
            residual: Vec::new(),
        };
        let mut derivative = CompleteDerivative::zero(1);
        derivative
            .add_iov_prior(&[1.0], &ndarray::array![[4.0]], &layout)
            .unwrap();
        derivative
            .add_iov_prior(&[2.0], &ndarray::array![[4.0]], &layout)
            .unwrap();
        assert!((derivative.score[0] + 3.0 / 32.0).abs() < 1e-12);
        assert!((derivative.hessian[[0, 0]] + 1.0 / 64.0).abs() < 1e-12);
    }

    #[test]
    fn multi_output_residual_coordinates_preserve_masks_and_order() {
        use crate::estimation::ParametricErrorModel;

        let models = ParametricErrorModels::new()
            .add(0, "first", ResidualErrorModel::constant(0.5).into())
            .add(
                1,
                "second",
                ParametricErrorModel::new(ResidualErrorModel::proportional(0.25)),
            );
        let empty = Array2::from_shape_vec((0, 0), Vec::<bool>::new()).unwrap();
        let layout = InformationLayout::new(
            &[],
            &[],
            &[],
            &[],
            &[],
            &empty,
            &empty,
            &[],
            None,
            None,
            &models,
        )
        .unwrap();
        assert_eq!(
            layout
                .coordinates
                .iter()
                .map(|coordinate| coordinate.name.as_str())
                .collect::<Vec<_>>(),
            ["residual:first:sigma", "residual:second:proportional"]
        );
        let mut derivative = CompleteDerivative::zero(2);
        derivative
            .add_residual(
                0,
                Some(2.0),
                1.0,
                Censor::None,
                ResidualErrorModel::constant(0.5),
                &layout,
            )
            .unwrap();
        derivative
            .add_residual(
                1,
                Some(3.0),
                2.0,
                Censor::None,
                ResidualErrorModel::proportional(0.25),
                &layout,
            )
            .unwrap();
        assert_eq!(derivative.score, vec![6.0, 12.0]);
        assert_eq!(derivative.hessian[[0, 0]], -44.0);
        assert_eq!(derivative.hessian[[1, 1]], -176.0);
        assert_eq!(derivative.hessian[[0, 1]], 0.0);
    }

    #[test]
    fn residual_family_derivatives_match_hard_coded_analytic_values() {
        let layout = residual_layout();
        for model in [
            ResidualErrorModel::constant(0.5),
            ResidualErrorModel::proportional(0.5),
        ] {
            let mut derivative = CompleteDerivative::zero(2);
            derivative
                .add_residual(0, Some(2.0), 1.0, Censor::None, model, &layout)
                .unwrap();
            let coordinate = if matches!(model, ResidualErrorModel::Proportional { .. }) {
                1
            } else {
                0
            };
            assert_eq!(derivative.score[coordinate], 6.0);
            assert_eq!(derivative.hessian[[coordinate, coordinate]], -44.0);
        }

        let mut combined = CompleteDerivative::zero(2);
        combined
            .add_residual(
                0,
                Some(2.0),
                1.0,
                Censor::None,
                ResidualErrorModel::combined(0.3, 0.4),
                &layout,
            )
            .unwrap();
        assert!((combined.score[0] - 3.6).abs() < 1e-12);
        assert!((combined.score[1] - 4.8).abs() < 1e-12);
        assert!((combined.hessian[[0, 0]] + 8.16).abs() < 1e-12);
        assert!((combined.hessian[[1, 1]] + 23.84).abs() < 1e-12);
        assert!((combined.hessian[[0, 1]] + 26.88).abs() < 1e-12);
        assert_eq!(combined.hessian[[0, 1]], combined.hessian[[1, 0]]);

        let mut exponential = CompleteDerivative::zero(2);
        exponential
            .add_residual(
                0,
                Some(std::f64::consts::E),
                1.0,
                Censor::None,
                ResidualErrorModel::exponential(0.5),
                &layout,
            )
            .unwrap();
        assert!((exponential.score[0] - 6.0).abs() < 1e-12);
        assert!((exponential.hessian[[0, 0]] + 44.0).abs() < 1e-12);
    }

    #[test]
    fn correlated_combined_layout_and_analytic_derivatives_match_finite_differences() {
        let models = ParametricErrorModels::new().add(
            0,
            "cp",
            crate::estimation::ParametricErrorModel::new(ResidualErrorModel::correlated_combined(
                0.6, 0.3, -0.25,
            )),
        );
        let empty = Array2::from_shape_vec((0, 0), Vec::<bool>::new()).unwrap();
        let layout = InformationLayout::new(
            &[],
            &[],
            &[],
            &[],
            &[],
            &empty,
            &empty,
            &[],
            None,
            None,
            &models,
        )
        .unwrap();
        assert_eq!(
            layout
                .coordinates
                .iter()
                .map(|coordinate| coordinate.name.as_str())
                .collect::<Vec<_>>(),
            [
                "residual:cp:additive",
                "residual:cp:proportional",
                "residual:cp:correlation"
            ]
        );

        let values = [0.6_f64, 0.3, -0.25];
        let prediction = 1.2_f64;
        let observation = 1.8_f64;
        let log_likelihood = |parameters: [f64; 3]| {
            let variance = parameters[0].powi(2)
                + 2.0 * parameters[2] * parameters[0] * parameters[1] * prediction
                + parameters[1].powi(2) * prediction.powi(2);
            -0.5 * (variance.ln() + (observation - prediction).powi(2) / variance)
        };
        let mut derivative = CompleteDerivative::zero(3);
        derivative
            .add_residual(
                0,
                Some(observation),
                prediction,
                Censor::None,
                ResidualErrorModel::correlated_combined(values[0], values[1], values[2]),
                &layout,
            )
            .unwrap();
        let h = 1e-4;
        for left in 0..3 {
            let mut plus = values;
            plus[left] += h;
            let mut minus = values;
            minus[left] -= h;
            let numeric_score = (log_likelihood(plus) - log_likelihood(minus)) / (2.0 * h);
            assert!((derivative.score[left] - numeric_score).abs() < 2e-7);
            for right in 0..3 {
                let numeric_hessian = if left == right {
                    (log_likelihood(plus) - 2.0 * log_likelihood(values) + log_likelihood(minus))
                        / h.powi(2)
                } else {
                    let mut pp = values;
                    pp[left] += h;
                    pp[right] += h;
                    let mut pm = values;
                    pm[left] += h;
                    pm[right] -= h;
                    let mut mp = values;
                    mp[left] -= h;
                    mp[right] += h;
                    let mut mm = values;
                    mm[left] -= h;
                    mm[right] -= h;
                    (log_likelihood(pp) - log_likelihood(pm) - log_likelihood(mp)
                        + log_likelihood(mm))
                        / (4.0 * h.powi(2))
                };
                assert!(
                    (derivative.hessian[[left, right]] - numeric_hessian).abs() < 2e-5,
                    "hessian ({left}, {right})"
                );
            }
        }
    }

    #[test]
    fn missing_and_censor_semantics_are_explicit() {
        let layout = residual_layout();
        let mut missing = CompleteDerivative::zero(2);
        missing
            .add_residual(
                0,
                None,
                1.0,
                Censor::None,
                ResidualErrorModel::constant(0.5),
                &layout,
            )
            .unwrap();
        assert_eq!(missing.score, vec![0.0, 0.0]);

        let mut censored = CompleteDerivative::zero(2);
        assert!(censored
            .add_residual(
                0,
                Some(1.0),
                1.0,
                Censor::BLOQ,
                ResidualErrorModel::constant(0.5),
                &layout
            )
            .unwrap_err()
            .to_string()
            .contains("censored"));
    }

    #[test]
    fn every_residual_family_matches_below_equal_and_above_likelihood_floor() {
        let layout = residual_layout();
        let floor = f64::EPSILON.sqrt();
        let branches = [floor / 2.0, floor, floor * 2.0];
        for family in ["constant", "proportional", "combined", "exponential"] {
            for (branch, scale) in branches.into_iter().enumerate() {
                let model = match family {
                    "constant" => ResidualErrorModel::constant(scale),
                    "proportional" => ResidualErrorModel::proportional(scale),
                    "combined" => ResidualErrorModel::combined(scale, 0.0),
                    "exponential" => ResidualErrorModel::exponential(scale),
                    _ => unreachable!(),
                };
                let mut derivative = CompleteDerivative::zero(2);
                let result =
                    derivative.add_residual(0, Some(1.0), 1.0, Censor::None, model, &layout);
                match branch {
                    0 => {
                        result.unwrap();
                        assert_eq!(derivative.score, vec![0.0, 0.0]);
                        assert_eq!(derivative.hessian, Array2::<f64>::zeros((2, 2)));
                    }
                    1 => assert_eq!(
                        result.unwrap_err().to_string(),
                        format!(
                            "{family} residual scale is exactly at the nondifferentiable likelihood floor boundary"
                        )
                    ),
                    2 => {
                        result.unwrap();
                        let coordinate = usize::from(family == "proportional");
                        assert_eq!(derivative.score[coordinate], -1.0 / scale);
                        assert_eq!(
                            derivative.hessian[[coordinate, coordinate]],
                            1.0 / scale.powi(2)
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }

        let mut prediction_floored = CompleteDerivative::zero(2);
        prediction_floored
            .add_residual(
                0,
                Some(1.0),
                0.0,
                Censor::None,
                ResidualErrorModel::proportional(0.5),
                &layout,
            )
            .unwrap();
        assert_eq!(prediction_floored.score, vec![0.0, 0.0]);
    }

    #[test]
    fn recursion_uses_mean_outer_scores_and_two_cycle_sa_updates() {
        let layout = one_coordinate_layout();
        let mut recursion = InformationRecursion::new(layout);
        let a = CompleteDerivative {
            score: vec![1.0],
            hessian: ndarray::array![[-2.0]],
        };
        let b = CompleteDerivative {
            score: vec![-1.0],
            hessian: ndarray::array![[-2.0]],
        };
        recursion.update(&[a, b], 1.0);
        let c = CompleteDerivative {
            score: vec![2.0],
            hessian: ndarray::array![[-4.0]],
        };
        recursion.update(&[c], 0.5);
        let diagnostics = recursion.diagnostics();
        assert_eq!(diagnostics.recursion_cycles, 2);
        assert_eq!(diagnostics.delta, vec![1.0]);
        assert_eq!(diagnostics.expected_complete_hessian, vec![vec![-3.0]]);
        assert_eq!(diagnostics.g, vec![vec![-0.5]]);
        assert_eq!(diagnostics.observed_hessian, vec![vec![-1.5]]);
        assert_eq!(diagnostics.observed_information, vec![vec![1.5]]);
        assert_eq!(diagnostics.status, InformationStatus::Available);
    }

    #[test]
    fn diagnostics_canonicalize_only_accepted_symmetric_roundoff() {
        let mut recursion = InformationRecursion::new(two_coordinate_layout());
        recursion.g = ndarray::array![[-4.0, -1.0], [-1.0 - f64::EPSILON, -3.0]];
        recursion.complete_hessian = ndarray::array![[-4.0, -1.0], [-1.0, -3.0]];

        let diagnostics = recursion.diagnostics();
        assert_eq!(diagnostics.status, InformationStatus::Available);
        assert_eq!(
            diagnostics.observed_information[0][1],
            diagnostics.observed_information[1][0]
        );
        assert_eq!(
            diagnostics.observed_hessian[0][1],
            diagnostics.observed_hessian[1][0]
        );
    }

    #[test]
    fn accepted_high_scale_roundoff_canonicalization_remains_finite() {
        let high = 0.25 * f64::MAX;
        let near = high * (1.0 + 8.0 * f64::EPSILON);
        let mut recursion = InformationRecursion::new(two_coordinate_layout());
        recursion.g = ndarray::array![[-3.0 * high, -high], [-near, -3.0 * high]];
        recursion.complete_hessian = ndarray::array![[-3.0 * high, -high], [-high, -3.0 * high]];

        let diagnostics = recursion.diagnostics();
        assert_eq!(diagnostics.status, InformationStatus::Available);
        assert!(diagnostics
            .observed_information
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
        assert_eq!(
            diagnostics.observed_information[0][1],
            diagnostics.observed_information[1][0]
        );
    }

    #[test]
    fn above_tolerance_asymmetry_is_retained_and_rejected() {
        let asymmetry = 128.0 * f64::EPSILON;
        let mut raw_g = InformationRecursion::new(two_coordinate_layout());
        raw_g.g = ndarray::array![[-4.0, -1.0], [-1.0 - asymmetry, -3.0]];
        raw_g.complete_hessian = ndarray::array![[-4.0, -1.0], [-1.0, -3.0]];
        let g_diagnostics = raw_g.diagnostics();
        assert_eq!(g_diagnostics.status, InformationStatus::NonFinite);
        assert_eq!(g_diagnostics.g[1][0], -1.0 - asymmetry);
        assert_eq!(g_diagnostics.observed_hessian[1][0], -1.0 - asymmetry);
        assert_eq!(g_diagnostics.observed_hessian[0][1], -1.0);

        let mut raw_complete = InformationRecursion::new(two_coordinate_layout());
        raw_complete.g = ndarray::array![[-4.0, -1.0], [-1.0, -3.0]];
        raw_complete.complete_hessian = ndarray::array![[-4.0, -1.0], [-1.0 - asymmetry, -3.0]];
        let complete_diagnostics = raw_complete.diagnostics();
        assert_eq!(complete_diagnostics.status, InformationStatus::NonFinite);
        assert_eq!(
            complete_diagnostics.expected_complete_hessian[1][0],
            -1.0 - asymmetry
        );
        assert_eq!(complete_diagnostics.observed_hessian[0][1], -1.0);
        assert_eq!(complete_diagnostics.observed_hessian[1][0], -1.0);
    }

    #[test]
    fn nonfinite_and_indefinite_statuses_retain_unmodified_values() {
        let mut nonfinite = InformationRecursion::new(one_coordinate_layout());
        nonfinite.mark_unavailable(InformationStatus::NonFinite);
        assert_eq!(nonfinite.diagnostics().status, InformationStatus::NonFinite);

        let mut indefinite = InformationRecursion::new(one_coordinate_layout());
        indefinite.update(
            &[CompleteDerivative {
                score: vec![0.0],
                hessian: ndarray::array![[1.0]],
            }],
            1.0,
        );
        let diagnostics = indefinite.diagnostics();
        assert_eq!(
            diagnostics.status,
            InformationStatus::ObservedInformationNotPositiveDefinite
        );
        assert_eq!(diagnostics.observed_information, vec![vec![-1.0]]);
    }

    #[test]
    fn zero_coordinates_are_labeled_without_fabricated_information() {
        let layout = InformationLayout {
            coordinates: Vec::new(),
            population: Vec::new(),
            covariate_effects: Vec::new(),
            omega: Vec::new(),
            omega_iov: Vec::new(),
            residual: Vec::new(),
        };
        assert_eq!(
            InformationRecursion::new(layout).diagnostics().status,
            InformationStatus::NoFreeCoordinates
        );
    }

    // ─── Covariate information coordinate tests ─────────────────────────

    fn covariate_effect_layout(
        covariate_names: &[String],
        covariate_estimated: &[bool],
    ) -> InformationLayout {
        let _empty = Array2::from_shape_vec((0, 0), Vec::<bool>::new()).unwrap();
        InformationLayout::new(
            &["CL".into()],
            &[true],
            covariate_names,
            covariate_estimated,
            &["CL".into()],
            &ndarray::array![[true]],
            &ndarray::array![[true]],
            &[],
            None,
            None,
            &crate::estimation::ParametricErrorModels::new(),
        )
        .unwrap()
    }

    #[test]
    fn covariate_coordinates_follow_intercepts_in_canonical_order() {
        let layout = covariate_effect_layout(&["beta:CL:WT".into()], &[true]);
        assert_eq!(
            layout
                .coordinates
                .iter()
                .map(|c| (c.name.as_str(), &c.kind))
                .collect::<Vec<_>>(),
            vec![
                (
                    "phi:CL",
                    &InformationCoordinateKind::Population { parameter_index: 0 }
                ),
                (
                    "beta:CL:WT",
                    &InformationCoordinateKind::CovariateEffect { effect_index: 0 }
                ),
                (
                    "omega:CL:CL",
                    &InformationCoordinateKind::Omega { row: 0, column: 0 }
                ),
            ]
        );
        assert_eq!(layout.len(), 3);
    }

    #[test]
    fn fixed_covariate_effects_are_excluded_from_coordinates() {
        let layout =
            covariate_effect_layout(&["beta:CL:WT".into(), "beta:CL:AGE".into()], &[false, true]);
        assert_eq!(
            layout
                .coordinates
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["phi:CL", "beta:CL:AGE", "omega:CL:CL"]
        );
        assert_eq!(layout.len(), 3);
    }

    #[test]
    fn design_mean_prior_matches_separate_add_gaussian_for_single_intercept() {
        // When the design matrix is an identity (one intercept per eta row),
        // add_design_mean_prior should match add_gaussian with the same mean
        // coordinate mapping.
        let layout = InformationLayout {
            coordinates: vec![
                InformationCoordinate {
                    index: 0,
                    name: "phi:CL".into(),
                    kind: InformationCoordinateKind::Population { parameter_index: 0 },
                },
                InformationCoordinate {
                    index: 1,
                    name: "omega:CL:CL".into(),
                    kind: InformationCoordinateKind::Omega { row: 0, column: 0 },
                },
            ],
            population: vec![Some(0)],
            covariate_effects: Vec::new(),
            omega: vec![CovarianceCoordinate {
                coordinate: 1,
                row: 0,
                column: 0,
            }],
            omega_iov: Vec::new(),
            residual: Vec::new(),
        };
        let covariance = ndarray::array![[4.0]];
        let eta = vec![2.0];

        // Via add_gaussian (established path)
        let mut ref_deriv = CompleteDerivative::zero(2);
        ref_deriv
            .add_gaussian(&eta, &covariance, &[Some(0)], &layout.omega)
            .unwrap();

        // Via add_design_mean_prior (identity design)
        let mut test_deriv = CompleteDerivative::zero(2);
        test_deriv
            .add_design_mean_prior(
                &eta,
                &covariance,
                &ndarray::array![[1.0]], // identity design for one intercept
                &[0],
                &layout.omega,
            )
            .unwrap();

        assert_eq!(ref_deriv.score, test_deriv.score);
        assert_eq!(ref_deriv.hessian, test_deriv.hessian);
    }

    #[test]
    fn design_mean_prior_beta_beta_block_is_exact() {
        // Two covariate effects on one eta row, no Omega coordinates.
        // design = [[x1, x2]] where x1=1.5, x2=-0.5
        // eta = 3.0, covariance = [[2.0]]
        // W = 0.5, z = 1.5
        // Score: score_1 = x1 * z = 2.25, score_2 = x2 * z = -0.75
        // Hessian: H_cc = -A' * W * A = -[[x1^2*W, x1*x2*W], [x2*x1*W, x2^2*W]]
        //       = -[[1.125, -0.375], [-0.375, 0.125]]
        let _layout = InformationLayout {
            coordinates: vec![
                InformationCoordinate {
                    index: 0,
                    name: "beta:CL:WT".into(),
                    kind: InformationCoordinateKind::CovariateEffect { effect_index: 0 },
                },
                InformationCoordinate {
                    index: 1,
                    name: "beta:CL:AGE".into(),
                    kind: InformationCoordinateKind::CovariateEffect { effect_index: 1 },
                },
            ],
            population: Vec::new(),
            covariate_effects: vec![Some(0), Some(1)],
            omega: Vec::new(),
            omega_iov: Vec::new(),
            residual: Vec::new(),
        };
        let mut derivative = CompleteDerivative::zero(2);
        derivative
            .add_design_mean_prior(
                &[3.0],
                &ndarray::array![[2.0]],
                &ndarray::array![[1.5, -0.5]],
                &[0, 1],
                &[],
            )
            .unwrap();
        assert!((derivative.score[0] - 2.25).abs() < 1e-12);
        assert!((derivative.score[1] + 0.75).abs() < 1e-12);
        assert!((derivative.hessian[[0, 0]] + 1.125).abs() < 1e-12);
        assert!((derivative.hessian[[0, 1]] - 0.375).abs() < 1e-12);
        assert_eq!(derivative.hessian[[0, 1]], derivative.hessian[[1, 0]]);
        assert!((derivative.hessian[[1, 1]] + 0.125).abs() < 1e-12);
    }

    #[test]
    fn design_mean_prior_beta_omega_cross_block_is_exact() {
        // One covariate effect and one Omega coordinate.
        // design = [[x]], eta = e, Omega = [[sigma^2]]
        // W = 1/sigma^2, z = e/sigma^2
        // S = [[1]]  (symmetric basis for diagonal)
        // W*S = W, W*S*z = z/sigma^2 = e/sigma^4
        // H_{beta,omega} = -x * (W*S*z) = -x * e / sigma^4
        let layout = InformationLayout {
            coordinates: vec![
                InformationCoordinate {
                    index: 0,
                    name: "beta:CL:WT".into(),
                    kind: InformationCoordinateKind::CovariateEffect { effect_index: 0 },
                },
                InformationCoordinate {
                    index: 1,
                    name: "omega:CL:CL".into(),
                    kind: InformationCoordinateKind::Omega { row: 0, column: 0 },
                },
            ],
            population: Vec::new(),
            covariate_effects: vec![Some(0)],
            omega: vec![CovarianceCoordinate {
                coordinate: 1,
                row: 0,
                column: 0,
            }],
            omega_iov: Vec::new(),
            residual: Vec::new(),
        };
        let covariance = ndarray::array![[4.0]]; // sigma^2 = 4, sigma = 2
        let eta = vec![3.0];
        let x = 1.5;
        let mut derivative = CompleteDerivative::zero(2);
        derivative
            .add_design_mean_prior(
                &eta,
                &covariance,
                &ndarray::array![[x]],
                &[0],
                &layout.omega,
            )
            .unwrap();
        // Score beta: x * z = x * eta / sigma^2 = 1.5 * 3/4 = 1.125
        assert!((derivative.score[0] - 1.125).abs() < 1e-12);
        // Score omega: -0.5*trace(W) + 0.5*eta'*W*S*W*eta
        //   = -0.5*(1/4) + 0.5 * 9/16 = -0.125 + 0.28125 = 0.15625...
        // Wait: quadratic = eta' * (W * S * z) = 3 * (1/4 * 1 * 3/4) = 3 * 3/16 = 9/16
        // score_omega = -0.5*trace(W*S) + 0.5*quadratic = -0.5*(1/4) + 0.5*9/16 = -0.125 + 0.28125
        let expected_omega_score = -0.125 + 0.28125;
        assert!((derivative.score[1] - expected_omega_score).abs() < 1e-12);
        // H_beta_beta = -x^2 * W = -2.25 * 0.25 = -0.5625
        assert!((derivative.hessian[[0, 0]] + 0.5625).abs() < 1e-12);
        // H_beta_omega = -x * W * S * z = -1.5 * 0.25 * 1 * 3/4 = -0.28125
        assert!((derivative.hessian[[0, 1]] + 0.28125).abs() < 1e-12);
        assert_eq!(derivative.hessian[[0, 1]], derivative.hessian[[1, 0]]);
    }

    #[test]
    fn design_mean_prior_matches_add_gaussian_for_iov_identity_design() {
        // IOV with identity design: kappa deviation and Omega_IOV.
        // add_design_mean_prior with the beta coordinate as a mean coordinate
        // should match add_gaussian with the same mapping.
        let omega_iov = vec![CovarianceCoordinate {
            coordinate: 1,
            row: 0,
            column: 0,
        }];
        let _layout = InformationLayout {
            coordinates: vec![
                InformationCoordinate {
                    index: 0,
                    name: "beta:CL:WT".into(),
                    kind: InformationCoordinateKind::CovariateEffect { effect_index: 0 },
                },
                InformationCoordinate {
                    index: 1,
                    name: "omega_iov:CL:CL".into(),
                    kind: InformationCoordinateKind::OmegaIov { row: 0, column: 0 },
                },
            ],
            population: Vec::new(),
            covariate_effects: vec![Some(0)],
            omega: Vec::new(),
            omega_iov: omega_iov.clone(),
            residual: Vec::new(),
        };
        let covariance = ndarray::array![[4.0]];
        let kappa = vec![2.0];

        // add_gaussian with Some(0) mean coordinate and omega_iov
        let mut ref_deriv = CompleteDerivative::zero(2);
        ref_deriv
            .add_gaussian(&kappa, &covariance, &[Some(0)], &omega_iov)
            .unwrap();

        // add_design_mean_prior with identity design matching the intercept mapping
        let mut test_deriv = CompleteDerivative::zero(2);
        test_deriv
            .add_design_mean_prior(
                &kappa,
                &covariance,
                &ndarray::array![[1.0]],
                &[0],
                &omega_iov,
            )
            .unwrap();

        assert_eq!(ref_deriv.score, test_deriv.score);
        assert_eq!(ref_deriv.hessian, test_deriv.hessian);
    }

    #[test]
    fn covariate_coordinate_kind_roundtrips_in_serde() {
        let kind = InformationCoordinateKind::CovariateEffect { effect_index: 7 };
        let json = serde_json::to_string(&kind).unwrap();
        assert!(json.contains("covariate_effect"));
        assert!(json.contains("7"));
        let roundtripped: InformationCoordinateKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped, kind);
    }
}
