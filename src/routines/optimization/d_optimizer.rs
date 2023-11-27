use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        CostFunction, Error, Executor,
    },
    solver::neldermead::NelderMead,
};
use ndarray::{Array1, Axis};

use crate::routines::{
    datafile::Scenario,
    simulation::predict::{sim_obs, Engine, Predict},
};

use crate::prelude::{prob, sigma::Sigma};

pub struct SppOptimizer<'a, S, P>
where
    S: Sigma + Sync,
    P: Predict + Sync + Clone,
{
    engine: &'a Engine<P>,
    scenarios: &'a Vec<Scenario>,
    sig: &'a S,
    pyl: &'a Array1<f64>,
}

impl<'a, S, P> CostFunction for SppOptimizer<'a, S, P>
where
    S: Sigma + Sync,
    P: Predict + Sync + Clone,
{
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = spp.to_owned().insert_axis(Axis(0));
        let ypred = sim_obs(&self.engine, &self.scenarios, &theta, true);
        let psi = prob::calculate_psi(&ypred, self.scenarios, self.sig);
        if psi.ncols() > 1 {
            log::error!("Psi in SppOptimizer has more than one column");
        }
        if psi.nrows() != self.pyl.len() {
            log::error!(
                "Psi in SppOptimizer has {} rows, but spp has {}",
                psi.nrows(),
                self.pyl.len()
            );
        }
        let nsub = psi.nrows() as f64;
        let mut sum = -nsub;
        for (p_i, pyl_i) in psi.iter().zip(self.pyl.iter()) {
            sum += nsub + p_i / pyl_i;
        }
        Ok(-sum)
    }
}

impl<'a, S, P> SppOptimizer<'a, S, P>
where
    S: Sigma + Sync,
    P: Predict + Sync + Clone,
{
    pub fn new(
        engine: &'a Engine<P>,
        scenarios: &'a Vec<Scenario>,
        sig: &'a S,
        pyl: &'a Array1<f64>,
    ) -> Self {
        Self {
            engine,
            scenarios,
            sig,
            pyl,
        }
    }
    pub fn optimize_point(self, spp: Array1<f64>) -> Result<Array1<f64>, Error> {
        let simplex = create_initial_simplex(&spp);
        let solver = NelderMead::new(simplex).with_sd_tolerance(1e-2)?;
        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(5))
            // .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;
        Ok(res.state.best_param.unwrap())
    }
}

fn create_initial_simplex(initial_point: &Array1<f64>) -> Vec<Array1<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.008;

    // Initialize a Vec to store the vertices of the simplex
    let mut vertices = Vec::new();

    // Add the initial point to the vertices
    vertices.push(initial_point.to_owned());

    // Calculate perturbation values for each component
    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025 // Special case for components equal to 0
        } else {
            perturbation_percentage * initial_point[i]
        };

        let mut perturbed_point = initial_point.to_owned();
        perturbed_point[i] += perturbation;
        vertices.push(perturbed_point);
    }

    vertices
}
