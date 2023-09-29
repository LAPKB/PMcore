use crate::prelude::{
    algorithms::Algorithm,
    datafile::Scenario,
    evaluation::sigma::{ErrorPoly, ErrorType},
    expansion, ipm,
    output::NPResult,
    output::{CycleLog, NPCycle},
    prob, qr,
    settings::run::Data,
    simulation::predict::Engine,
    simulation::predict::{sim_obs, Predict},
};

use ndarray::{stack, Array, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_stats::{DeviationExt, QuantileExt};
use tokio::sync::mpsc::UnboundedSender;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub struct POSTPROB<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    engine: Engine<S>,
    ranges: Vec<(f64, f64)>,
    psi: Array2<f64>,
    theta: Array2<f64>,
    lambda: Array1<f64>,
    w: Array1<f64>,
    objf: f64,
    cycle: usize,
    converged: bool,
    gamma: f64,
    error_type: ErrorType,
    scenarios: Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<NPCycle>,
    settings: Data,
}

impl<S> Algorithm<S> for POSTPROB<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    fn fit(&mut self) -> (Engine<S>, NPResult) {
        self.run()
    }
}

impl<S> POSTPROB<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    pub fn new(
        sim_eng: Engine<S>,
        ranges: Vec<(f64, f64)>,
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
            ranges,
            psi: Array2::default((0, 0)),
            theta,
            lambda: Array1::default(0),
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

    pub fn run(&mut self) -> (Engine<S>, NPResult) {
        
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

        return(
            self.engine.clone(),
            NPResult::new(
                self.scenarios.clone(),
                self.theta.clone(),
                self.psi.clone(),
                self.w.clone(),
                self.objf,
                self.cycle,
                self.converged,
                self.settings.clone(),
            ),
        )
    }
}