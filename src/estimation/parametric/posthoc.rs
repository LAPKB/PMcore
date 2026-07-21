use argmin::{
    core::{CostFunction, Error, Executor, State, TerminationReason},
    solver::neldermead::NelderMead,
};

const NON_FINITE_PENALTY: f64 = 1e100;

#[derive(Debug, Clone)]
pub(crate) struct ConditionalModeSolution {
    pub(crate) coordinates: Vec<f64>,
    pub(crate) objective: f64,
    pub(crate) converged: bool,
    pub(crate) iterations: u64,
    pub(crate) termination: String,
}

struct ConditionalModeCost<F> {
    cost: F,
}

impl<F> CostFunction for ConditionalModeCost<F>
where
    F: Fn(&[f64]) -> f64,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, coordinates: &Self::Param) -> Result<Self::Output, Error> {
        let objective = (self.cost)(coordinates);
        Ok(if objective.is_finite() {
            objective
        } else {
            NON_FINITE_PENALTY
        })
    }
}

pub(crate) fn optimize_conditional_mode<F>(
    initial: Vec<f64>,
    coordinate_scales: &[f64],
    max_iterations: u64,
    sd_tolerance: f64,
    cost: F,
) -> anyhow::Result<ConditionalModeSolution>
where
    F: Fn(&[f64]) -> f64,
{
    anyhow::ensure!(
        !initial.is_empty(),
        "conditional mode requires latent coordinates"
    );
    anyhow::ensure!(
        initial.len() == coordinate_scales.len(),
        "conditional-mode coordinate and scale dimensions differ"
    );
    anyhow::ensure!(
        initial.iter().all(|value| value.is_finite()),
        "conditional-mode initial coordinates must be finite"
    );
    anyhow::ensure!(
        coordinate_scales
            .iter()
            .all(|scale| scale.is_finite() && *scale > 0.0),
        "conditional-mode coordinate scales must be finite and positive"
    );
    anyhow::ensure!(
        sd_tolerance.is_finite() && sd_tolerance > 0.0,
        "conditional-mode tolerance must be finite and positive"
    );

    let initial_objective = cost(&initial);
    anyhow::ensure!(
        initial_objective.is_finite(),
        "conditional-mode warm-start objective is non-finite"
    );

    let solver = NelderMead::new(initial_simplex(&initial, coordinate_scales))
        .with_sd_tolerance(sd_tolerance)?;
    let result = Executor::new(ConditionalModeCost { cost }, solver)
        .configure(|state| state.max_iters(max_iterations))
        .run()?;
    let state = result.state;
    let coordinates = state
        .best_param
        .clone()
        .filter(|_coordinates| state.best_cost.is_finite() && state.best_cost < NON_FINITE_PENALTY)
        .unwrap_or_else(|| initial.clone());
    let objective = if state.best_cost.is_finite() && state.best_cost < NON_FINITE_PENALTY {
        state.best_cost
    } else {
        initial_objective
    };
    let termination_reason = state.get_termination_reason();
    let converged = matches!(termination_reason, Some(TerminationReason::SolverConverged));
    let termination = termination_reason
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown termination".to_owned());

    Ok(ConditionalModeSolution {
        coordinates,
        objective,
        converged,
        iterations: state.iter,
        termination,
    })
}

fn initial_simplex(initial: &[f64], coordinate_scales: &[f64]) -> Vec<Vec<f64>> {
    let mut simplex = Vec::with_capacity(initial.len() + 1);
    simplex.push(initial.to_vec());
    for dimension in 0..initial.len() {
        let mut point = initial.to_vec();
        point[dimension] += coordinate_scales[dimension];
        simplex.push(point);
    }
    simplex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_recovers_known_joint_mode() {
        let solution =
            optimize_conditional_mode(vec![0.0, 0.0], &[0.2, 0.2], 200, 1e-10, |coordinates| {
                (coordinates[0] - 1.0).powi(2) + 2.0 * (coordinates[1] + 0.5).powi(2)
            })
            .unwrap();

        assert!((solution.coordinates[0] - 1.0).abs() < 1e-4);
        assert!((solution.coordinates[1] + 0.5).abs() < 1e-4);
        assert!(solution.objective < 1e-8);
        assert!(solution.converged);
    }

    #[test]
    fn non_finite_regions_receive_finite_penalty() {
        let solution = optimize_conditional_mode(vec![0.0], &[0.1], 100, 1e-8, |coordinates| {
            if coordinates[0] > 0.5 {
                f64::NAN
            } else {
                (coordinates[0] - 0.25).powi(2)
            }
        })
        .unwrap();

        assert!((solution.coordinates[0] - 0.25).abs() < 1e-3);
        assert!(solution.objective.is_finite());
    }

    #[test]
    fn non_finite_warm_start_is_rejected() {
        let error =
            optimize_conditional_mode(vec![0.0], &[0.1], 10, 1e-6, |_| f64::INFINITY).unwrap_err();
        assert!(error.to_string().contains("warm-start objective"));
    }
}
