use std::path::Path;

use pharmsol::Equation;
use serde::Serialize;

use crate::algorithms::Status;
use crate::estimation::nonparametric::{CycleLog, NPPredictions, Posterior, Psi, Theta, Weights};

use pharmsol::{AssayErrorModels, Data};

/// Contains the results of a nonparametric estimation, including the final parameter
#[derive(Debug)]
pub struct NonParametricResult<E: Equation> {
    equation: E,
    data: Data,
    error_models: AssayErrorModels,
    prior: Theta,
    theta: Theta,
    psi: Psi,
    weights: Weights,
    objf: f64,
    cycles: usize,
    status: Status,
    cyclelog: CycleLog,
}

impl<E: Equation> NonParametricResult<E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        prior: Theta,
        theta: Theta,
        psi: Psi,
        weights: Weights,
        objf: f64,
        cycles: usize,
        status: Status,
        cyclelog: CycleLog,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            equation,
            data,
            error_models,
            prior,
            theta,
            psi,
            weights,
            objf,
            cycles,
            status,
            cyclelog,
        })
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn objf(&self) -> f64 {
        self.objf
    }

    pub fn converged(&self) -> bool {
        self.status.converged()
    }

    pub fn get_theta(&self) -> &Theta {
        &self.theta
    }

    /// The prior distribution ([`Theta`]) that seeded the algorithm.
    ///
    /// This is the initial set of support points, as opposed to the optimized
    /// solution returned by [`get_theta`](Self::get_theta).
    pub fn prior(&self) -> &Theta {
        &self.prior
    }

    pub fn data(&self) -> &Data {
        &self.data
    }

    pub fn equation(&self) -> &E {
        &self.equation
    }

    pub fn cycle_log(&self) -> &CycleLog {
        &self.cyclelog
    }

    /// Chain this result into a new fit with a different algorithm.
    ///
    /// Uses the optimized theta as the prior for the new run. Keeps the same
    /// equation, data, and error models. Consumes `self`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = problem
    ///     .fit_with(NpagConfig::new().max_cycles(100))?
    ///     .chain(NpodConfig::new().max_cycles(50))?;
    /// ```
    pub fn chain<A>(self, algorithm: A) -> anyhow::Result<NonParametricResult<E>>
    where
        A: crate::algorithms::Algorithm<
            E,
            crate::estimation::NonParametric,
            Output = NonParametricResult<E>,
        >,
        E: crate::model::EquationMetadataSource,
    {
        crate::estimation::EstimationProblem::nonparametric(
            self.equation,
            self.data,
            self.theta,
            self.error_models,
        )?
        .fit_with(algorithm)
    }

    pub fn psi(&self) -> &Psi {
        &self.psi
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    /// Compute the posterior probabilities on demand from [`Psi`] and the
    /// [`Weights`]. This is a cheap matrix operation and is intentionally not
    /// cached on the result.
    pub fn posterior(&self) -> anyhow::Result<Posterior> {
        Posterior::calculate(&self.psi, &self.weights)
    }

    /// Compute predictions on demand. Nothing is cached on the result; callers
    /// that need the predictions repeatedly should hold on to the returned
    /// value themselves.
    pub fn predictions(&self, idelta: f64, tad: f64) -> anyhow::Result<NPPredictions> {
        let posterior = self.posterior()?;
        self.predictions_with(&posterior, idelta, tad)
    }

    /// Like [`predictions`](Self::predictions), but reuses an already-computed
    /// [`Posterior`] instead of recomputing it.
    fn predictions_with(
        &self,
        posterior: &Posterior,
        idelta: f64,
        tad: f64,
    ) -> anyhow::Result<NPPredictions> {
        NPPredictions::calculate(
            &self.equation,
            &self.data,
            &self.theta,
            &self.weights,
            posterior,
            idelta,
            tad,
        )
    }

    pub fn write_theta(&self, path: &Path) -> anyhow::Result<()> {
        use anyhow::bail;
        use csv::WriterBuilder;

        tracing::debug!("Writing population parameter distribution...");

        let w: Vec<f64> = self.weights.to_vec();
        if w.len() != self.theta.matrix().nrows() {
            bail!(
                "Number of weights ({}) and number of support points ({}) do not match.",
                w.len(),
                self.theta.matrix().nrows()
            );
        }

        crate::estimation::nonparametric::create_parent_dir(path)?;
        let file = std::fs::File::create(path)?;
        let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

        let mut theta_header = self.theta.param_names().clone();
        theta_header.push("prob".to_string());
        writer.write_record(&theta_header)?;

        for (theta_row, &w_val) in self.theta.matrix().row_iter().zip(w.iter()) {
            let mut row: Vec<String> = theta_row.iter().map(|&val| val.to_string()).collect();
            row.push(w_val.to_string());
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_posterior(&self, path: &Path) -> anyhow::Result<()> {
        let posterior = self.posterior()?;
        self.write_posterior_with(path, &posterior)
    }

    fn write_posterior_with(&self, path: &Path, posterior: &Posterior) -> anyhow::Result<()> {
        use csv::WriterBuilder;

        tracing::debug!("Writing posterior parameter probabilities...");

        crate::estimation::nonparametric::create_parent_dir(path)?;
        let file = std::fs::File::create(path)?;
        let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

        writer.write_field("id")?;
        writer.write_field("point")?;
        self.theta.param_names().iter().for_each(|name| {
            writer.write_field(name).unwrap();
        });
        writer.write_field("prob")?;
        writer.write_record(None::<&[u8]>)?;

        let subjects = self.data.subjects();
        posterior
            .matrix()
            .row_iter()
            .enumerate()
            .for_each(|(i, row)| {
                let subject = subjects.get(i).unwrap();
                let id = subject.id();

                row.iter().enumerate().for_each(|(spp, prob)| {
                    writer.write_field(id.clone()).unwrap();
                    writer.write_field(spp.to_string()).unwrap();

                    self.theta.matrix().row(spp).iter().for_each(|val| {
                        writer.write_field(val.to_string()).unwrap();
                    });

                    writer.write_field(prob.to_string()).unwrap();
                    writer.write_record(None::<&[u8]>).unwrap();
                });
            });

        writer.flush()?;
        Ok(())
    }

    pub fn write_covariates(&self, path: &Path) -> anyhow::Result<()> {
        use csv::WriterBuilder;
        use pharmsol::Event;

        tracing::debug!("Writing covariates...");
        crate::estimation::nonparametric::create_parent_dir(path)?;
        let file = std::fs::File::create(path)?;
        let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

        let mut covariate_names = std::collections::HashSet::new();
        for subject in self.data.subjects() {
            for occasion in subject.occasions() {
                let covmap = occasion.covariates().covariates();
                for cov_name in covmap.keys() {
                    covariate_names.insert(cov_name.clone());
                }
            }
        }
        let mut covariate_names: Vec<String> = covariate_names.into_iter().collect();
        covariate_names.sort();

        let mut headers = vec!["id", "time", "block"];
        headers.extend(covariate_names.iter().map(|s| s.as_str()));
        writer.write_record(&headers)?;

        for subject in self.data.subjects() {
            for occasion in subject.occasions() {
                let covmap = occasion.covariates().covariates();

                for event in occasion.iter() {
                    let time = match event {
                        Event::Bolus(bolus) => bolus.time(),
                        Event::Infusion(infusion) => infusion.time(),
                        Event::Observation(observation) => observation.time(),
                    };

                    let mut row: Vec<String> = Vec::new();
                    row.push(subject.id().clone());
                    row.push(time.to_string());
                    row.push(occasion.index().to_string());

                    for cov_name in &covariate_names {
                        if let Some(cov) = covmap.get(cov_name) {
                            if let Ok(value) = cov.interpolate(time) {
                                row.push(value.to_string());
                            } else {
                                row.push(String::new());
                            }
                        } else {
                            row.push(String::new());
                        }
                    }

                    writer.write_record(&row)?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Write the cycle log to a CSV file readable by Pmetrics.
    pub fn write_cycles(&self, path: &Path) -> anyhow::Result<()> {
        self.cyclelog.write(path)
    }

    /// Compute and write the population and posterior predictions to a CSV file
    /// readable by Pmetrics.
    ///
    /// `idelta` is the interval used to densify the prediction grid and `tad` is
    /// the additional time after the last event to simulate.
    pub fn write_predictions(&self, path: &Path, idelta: f64, tad: f64) -> anyhow::Result<()> {
        let predictions = self.predictions(idelta, tad)?;
        predictions.write(path)
    }

    /// Serialize the complete result to a single JSON file.
    ///
    /// This includes the data, error models, prior, optimized support points,
    /// likelihoods, weights, objective function, status, cycle log, posterior
    /// probabilities, and predictions. `idelta` and `tad` control the density of
    /// the embedded predictions (see [`predictions`](Self::predictions)).
    pub fn write_json(&self, path: &Path, idelta: f64, tad: f64) -> anyhow::Result<()> {
        let posterior = self.posterior()?;
        let predictions = self.predictions_with(&posterior, idelta, tad)?;
        self.write_json_with(path, &posterior, &predictions)
    }

    fn write_json_with(
        &self,
        path: &Path,
        posterior: &Posterior,
        predictions: &NPPredictions,
    ) -> anyhow::Result<()> {
        tracing::debug!("Writing result as JSON...");

        crate::estimation::nonparametric::create_parent_dir(path)?;

        let view = NonParametricResultJson {
            data: &self.data,
            error_models: &self.error_models,
            prior: &self.prior,
            theta: &self.theta,
            psi: &self.psi,
            weights: &self.weights,
            objf: self.objf,
            cycles: self.cycles,
            status: &self.status,
            cyclelog: &self.cyclelog,
            posterior,
            predictions,
        };

        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &view)?;
        Ok(())
    }

    /// Write the full set of result artifacts to `directory`.
    ///
    /// The directory is created if it does not exist, and the following files
    /// are produced (matching the names expected by Pmetrics):
    /// - `theta.csv` — optimized support points with weights
    /// - `posterior.csv` — per-subject posterior probabilities
    /// - `pred.csv` — population and posterior predictions
    /// - `covs.csv` — interpolated covariates
    /// - `cycles.csv` — cycle log
    /// - `result.json` — the complete result as JSON
    ///
    /// The posterior and predictions are computed once and shared across the CSV
    /// and JSON outputs. `idelta` and `tad` control the density of the
    /// predictions (see [`predictions`](Self::predictions)).
    pub fn write_outputs(
        &self,
        directory: impl AsRef<Path>,
        idelta: f64,
        tad: f64,
    ) -> anyhow::Result<()> {
        let dir = directory.as_ref();
        std::fs::create_dir_all(dir)?;

        // Compute the expensive artifacts a single time and reuse them.
        let posterior = self.posterior()?;
        let predictions = self.predictions_with(&posterior, idelta, tad)?;

        self.write_theta(&dir.join("theta.csv"))?;
        self.write_posterior_with(&dir.join("posterior.csv"), &posterior)?;
        predictions.write(&dir.join("pred.csv"))?;
        self.write_covariates(&dir.join("covs.csv"))?;
        self.write_cycles(&dir.join("cycles.csv"))?;
        self.write_json_with(&dir.join("result.json"), &posterior, &predictions)?;

        tracing::info!("Results written to {}", dir.display());
        Ok(())
    }
}

/// A borrowed, serializable view over [`NonParametricResult`] used to emit a
/// single JSON document. The equation is intentionally omitted because it is
/// generic and not necessarily serializable.
#[derive(Serialize)]
struct NonParametricResultJson<'a> {
    data: &'a Data,
    error_models: &'a AssayErrorModels,
    prior: &'a Theta,
    theta: &'a Theta,
    psi: &'a Psi,
    weights: &'a Weights,
    objf: f64,
    cycles: usize,
    status: &'a Status,
    cyclelog: &'a CycleLog,
    posterior: &'a Posterior,
    predictions: &'a NPPredictions,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::nonparametric::{NpagConfig, NpodConfig};
    use crate::estimation::EstimationProblem;
    use crate::model::ParameterSpace;
    use pharmsol::equation::metadata;
    use pharmsol::prelude::data::{AssayErrorModel, AssayErrorModels};
    use pharmsol::{ErrorPoly, SubjectBuilderExt};

    fn minimal_ode() -> pharmsol::ODE {
        pharmsol::equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                let ke = p[0];
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| pharmsol::lag! {},
            |_p, _t, _cov| pharmsol::fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                let v = p[1];
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            metadata::new("chain_test")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["0"])
                .route(pharmsol::equation::Route::bolus("0").to_state("central")),
        )
        .expect("metadata should validate")
    }

    fn minimal_data() -> pharmsol::Data {
        let subject = pharmsol::Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .build();
        pharmsol::Data::new(vec![subject])
    }

    #[test]
    fn chain_npag_to_npod_preserves_support_points() {
        let ode = minimal_ode();
        let data = minimal_data();
        let params = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let prior = Theta::sobol_default(&params).unwrap();
        let err = AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
            )
            .unwrap();

        let r1 = EstimationProblem::nonparametric(ode, data, prior, err)
            .unwrap()
            .fit_with(NpagConfig::new().max_cycles(2))
            .unwrap();

        let n_spp = r1.get_theta().nspp();
        assert!(n_spp > 0, "NPAG should produce support points");

        // Chain into NPOD — should complete without error
        let r2 = r1.chain(NpodConfig::new().max_cycles(1)).unwrap();

        assert!(r2.get_theta().nspp() > 0);
        assert_eq!(r2.data().subjects().len(), 1);
        assert_eq!(r2.cycles(), 1);
    }

    #[test]
    fn chain_npag_to_npag_maintains_or_improves_objf() {
        let ode = minimal_ode();
        let data = minimal_data();
        let params = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let prior = Theta::sobol_with_seed(&params, 5, 42).unwrap();
        let err = AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
            )
            .unwrap();

        let r1 = EstimationProblem::nonparametric(ode, data, prior, err)
            .unwrap()
            .fit_with(NpagConfig::new().max_cycles(5))
            .unwrap();

        let objf1 = r1.objf();

        // Chain into another NPAG run
        let r2 = r1.chain(NpagConfig::new().max_cycles(3)).unwrap();

        // Second run should not regress significantly
        assert!(
            r2.objf() <= objf1 + 0.5,
            "OBJF regressed: {} -> {}",
            objf1,
            r2.objf()
        );
    }

    #[test]
    fn chain_npag_to_npod_with_unconverged_result() {
        let ode = minimal_ode();
        let data = minimal_data();
        let params = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let prior = Theta::sobol_with_seed(&params, 5, 42).unwrap();
        let err = AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
            )
            .unwrap();

        // Run only 1 cycle — will be unconverged
        let r1 = EstimationProblem::nonparametric(ode, data, prior, err)
            .unwrap()
            .fit_with(NpagConfig::new().max_cycles(1))
            .unwrap();

        // Chaining from unconverged should still work
        let r2 = r1.chain(NpodConfig::new().max_cycles(1));
        assert!(
            r2.is_ok(),
            "Chaining from unconverged result should succeed"
        );
    }
}
