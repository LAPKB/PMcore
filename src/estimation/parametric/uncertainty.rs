use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use pharmsol::Equation;

use super::{FimMethod, ParametricWorkspace, Population, UncertaintyEstimates};

impl UncertaintyEstimates {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_fim(&self) -> bool {
        self.fim.is_some()
    }

    pub fn has_standard_errors(&self) -> bool {
        self.se_mu.is_some()
    }

    pub fn fim_inverse(&self) -> Option<&Mat<f64>> {
        self.fim_inverse.as_ref()
    }

    pub fn se_mu(&self) -> Option<&Col<f64>> {
        self.se_mu.as_ref()
    }

    pub fn se_omega(&self) -> Option<&Mat<f64>> {
        self.se_omega.as_ref()
    }

    pub fn rse_mu(&self) -> Option<&Col<f64>> {
        self.rse_mu.as_ref()
    }

    pub fn fim_method(&self) -> Option<FimMethod> {
        self.fim_method
    }

    pub fn from_fim_inverse(
        population: &Population,
        fim_inverse: Mat<f64>,
        fim_method: FimMethod,
    ) -> Self {
        let n_params = population.npar();
        let se_mu =
            (fim_inverse.nrows() >= n_params && fim_inverse.ncols() >= n_params).then(|| {
                Col::from_fn(n_params, |index| {
                    fim_inverse[(index, index)].max(0.0).sqrt()
                })
            });
        let rse_mu = se_mu.as_ref().map(|se| {
            Col::from_fn(n_params, |index| {
                let mu = population.mu()[index].abs();
                if mu > f64::EPSILON {
                    100.0 * se[index] / mu
                } else {
                    0.0
                }
            })
        });
        let se_omega = if fim_inverse.nrows() >= 2 * n_params && fim_inverse.ncols() >= 2 * n_params
        {
            Some(Mat::from_fn(n_params, n_params, |row, col| {
                if row == col {
                    fim_inverse[(n_params + row, n_params + col)]
                        .max(0.0)
                        .sqrt()
                } else {
                    0.0
                }
            }))
        } else {
            None
        };

        Self {
            fim: None,
            fim_inverse: Some(fim_inverse),
            se_mu,
            se_omega,
            rse_mu,
            fim_method: Some(fim_method),
        }
    }

    pub fn from_fim(population: &Population, fim: Mat<f64>, fim_method: FimMethod) -> Result<Self> {
        let fim_inverse = fim
            .clone()
            .llt(faer::Side::Lower)
            .map_err(|_| anyhow::anyhow!("FIM is not positive definite"))?
            .inverse();
        let mut estimates = Self::from_fim_inverse(population, fim_inverse, fim_method);
        estimates.fim = Some(fim);
        Ok(estimates)
    }
}

pub fn focei_linearization_uncertainty(
    population: &Population,
    n_subjects: usize,
) -> UncertaintyEstimates {
    let n_params = population.npar();
    if n_subjects == 0 || n_params == 0 {
        return UncertaintyEstimates::new();
    }

    let n_subjects = n_subjects as f64;
    let omega_inv = inverse_or_diagonal(population.omega());
    let fim = Mat::from_fn(2 * n_params, 2 * n_params, |row, col| {
        if row < n_params && col < n_params {
            n_subjects * omega_inv[(row, col)]
        } else if row == col && row >= n_params {
            let variance = population.omega()[(row - n_params, row - n_params)].max(1e-8);
            n_subjects / (2.0 * variance.powi(2))
        } else {
            0.0
        }
    });

    UncertaintyEstimates::from_fim(population, fim, FimMethod::Linearization).unwrap_or_else(|_| {
        let fim_inverse = Mat::from_fn(2 * n_params, 2 * n_params, |row, col| {
            if row < n_params && col < n_params {
                population.omega()[(row, col)] / n_subjects
            } else if row == col && row >= n_params {
                let variance = population.omega()[(row - n_params, row - n_params)].max(1e-8);
                2.0 * variance.powi(2) / n_subjects
            } else {
                0.0
            }
        });
        UncertaintyEstimates::from_fim_inverse(population, fim_inverse, FimMethod::Linearization)
    })
}

fn inverse_or_diagonal(matrix: &Mat<f64>) -> Mat<f64> {
    match matrix.clone().llt(faer::Side::Lower) {
        Ok(cholesky) => cholesky.inverse(),
        Err(_) => Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
            if row == col {
                1.0 / matrix[(row, row)].max(1e-8)
            } else {
                0.0
            }
        }),
    }
}

pub fn estimates<E: Equation>(workspace: &ParametricWorkspace<E>) -> &UncertaintyEstimates {
    workspace.uncertainty()
}

pub fn has_fim<E: Equation>(workspace: &ParametricWorkspace<E>) -> bool {
    workspace.uncertainty().has_fim()
}

pub fn has_standard_errors<E: Equation>(workspace: &ParametricWorkspace<E>) -> bool {
    workspace.uncertainty().has_standard_errors()
}

pub fn se_mu<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Col<f64>> {
    workspace.uncertainty().se_mu()
}

pub fn fim<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Mat<f64>> {
    workspace.uncertainty().fim.as_ref()
}

pub fn fim_inverse<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Mat<f64>> {
    workspace.uncertainty().fim_inverse()
}

pub fn se_omega<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Mat<f64>> {
    workspace.uncertainty().se_omega()
}

pub fn rse_mu<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<&Col<f64>> {
    workspace.uncertainty().rse_mu()
}

pub fn fim_method<E: Equation>(workspace: &ParametricWorkspace<E>) -> Option<FimMethod> {
    workspace.uncertainty().fim_method()
}

#[cfg(test)]
mod tests {
    use super::{focei_linearization_uncertainty, UncertaintyEstimates};
    use crate::estimation::parametric::{FimMethod, Population};
    use crate::model::{ParameterSpace, ParameterSpec};
    use faer::{Col, Mat};

    #[test]
    fn derives_standard_errors_from_inverse_fim() {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 1.0, 20.0));
        let population = Population::new(
            Col::from_fn(2, |index| if index == 0 { 0.5 } else { 10.0 }),
            Mat::from_fn(2, 2, |row, col| if row == col { 0.2 } else { 0.0 }),
            parameters,
        )
        .unwrap();
        let fim_inverse = Mat::from_fn(
            4,
            4,
            |row, col| if row == col { (row + 1) as f64 } else { 0.0 },
        );

        let estimates =
            UncertaintyEstimates::from_fim_inverse(&population, fim_inverse, FimMethod::Observed);

        assert!(estimates.has_standard_errors());
        assert_eq!(estimates.fim_method(), Some(FimMethod::Observed));
        assert!((estimates.se_mu().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((estimates.se_mu().unwrap()[1] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((estimates.rse_mu().unwrap()[0] - 200.0).abs() < 1e-12);
        assert!(estimates.se_omega().is_some());
    }

    #[test]
    fn focei_linearization_uncertainty_exposes_fim_and_standard_errors() {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 1.0, 20.0));
        let population = Population::new(
            Col::from_fn(2, |index| if index == 0 { 0.5 } else { 10.0 }),
            Mat::from_fn(2, 2, |row, col| if row == col { 0.2 } else { 0.0 }),
            parameters,
        )
        .unwrap();

        let estimates = focei_linearization_uncertainty(&population, 4);

        assert!(estimates.has_fim());
        assert!(estimates.has_standard_errors());
        assert_eq!(estimates.fim_method(), Some(FimMethod::Linearization));
        assert!(estimates.fim_inverse().is_some());
    }
}
