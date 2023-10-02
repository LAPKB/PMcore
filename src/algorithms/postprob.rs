use crate::prelude::{
    algorithms::Algorithm,
    datafile::Scenario,
    evaluation::sigma::{ErrorPoly, ErrorType},
    ipm,
    output::NPCycle,
    output::NPResult,
    prob,
    settings::run::Data,
    simulation::predict::Engine,
    simulation::predict::{sim_obs, Predict},
};

use ndarray::{Array1, Array2};
use tokio::sync::mpsc::UnboundedSender;

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct POSTPROB<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    engine: Engine<S>,
    psi: Array2<f64>,
    theta: Array2<f64>,
    w: Array1<f64>,
    objf: f64,
    cycle: usize,
    converged: bool,
    gamma: f64,
    error_type: ErrorType,
    scenarios: Vec<Scenario>,
    c: (f64, f64, f64, f64),
    #[allow(dead_code)]
    tx: UnboundedSender<NPCycle>,
    settings: Data,
}

impl<S> Algorithm<S> for POSTPROB<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    fn fit(&mut self) -> NPResult {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
            self.scenarios.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.converged,
            self.settings.clone(),
        )
    }
}

impl<S> POSTPROB<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    pub fn new(
        sim_eng: Engine<S>,
        theta: Array2<f64>,
        scenarios: Vec<Scenario>,
        c: (f64, f64, f64, f64),
        tx: UnboundedSender<NPCycle>,
        settings: Data,
    ) -> Self
    where
        S: Predict + std::marker::Sync,
    {
        Self {
            engine: sim_eng,
            psi: Array2::default((0, 0)),
            theta,
            w: Array1::default(0),
            objf: f64::INFINITY,
            cycle: 0,
            converged: false,
            gamma: settings.parsed.error.value,
            error_type: match settings.parsed.error.class.as_str() {
                "additive" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            tx,
            settings,
            scenarios,
            c,
        }
    }

    pub fn run(&mut self) -> NPResult {
        let ypred = sim_obs(&self.engine, &self.scenarios, &self.theta, false);

        self.psi = prob::calculate_psi(
            &ypred,
            &self.scenarios,
            &ErrorPoly {
                c: self.c,
                gl: self.gamma,
                e_type: &self.error_type,
            },
        );

        let (w, objf) = ipm::burke(&self.psi).expect("Error in IPM");
        self.w = w;
        self.objf = objf;

        self.to_npresult()
    }
}
