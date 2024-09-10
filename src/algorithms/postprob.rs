use pharmsol::prelude::{
    data::{Data, ErrorModel, ErrorType},
    simulator::{psi, Equation},
};

use crate::{
    prelude::{algorithms::Algorithm, ipm::burke, output::NPResult, settings::Settings},
    tui::ui::Comm,
};

use ndarray::{Array1, Array2};
use tokio::sync::mpsc::UnboundedSender;

use super::output::CycleLog;

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct POSTPROB<E: Equation> {
    equation: E,
    psi: Array2<f64>,
    theta: Array2<f64>,
    w: Array1<f64>,
    objf: f64,
    cycle: usize,
    converged: bool,
    gamma: f64,
    error_type: ErrorType,
    data: Data,
    c: (f64, f64, f64, f64),
    #[allow(dead_code)]
    tx: Option<UnboundedSender<Comm>>,
    settings: Settings,
    cyclelog: CycleLog,
}

impl<E: Equation> Algorithm for POSTPROB<E> {
    fn fit(&mut self) -> anyhow::Result<NPResult, (anyhow::Error, NPResult)> {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.converged,
            self.settings.clone(),
            self.cyclelog.clone(),
        )
    }
}

impl<E: Equation> POSTPROB<E> {
    pub fn new(
        equation: E,
        theta: Array2<f64>,
        data: Data,
        c: (f64, f64, f64, f64),
        tx: Option<UnboundedSender<Comm>>,
        settings: Settings,
    ) -> Self {
        Self {
            equation,
            psi: Array2::default((0, 0)),
            theta,
            w: Array1::default(0),
            objf: f64::INFINITY,
            cycle: 0,
            converged: false,
            gamma: settings.error.value,
            error_type: match settings.error.class.as_str() {
                "additive" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            tx,
            settings,
            data,
            c,
            cyclelog: CycleLog::new(),
        }
    }

    pub fn run(&mut self) -> anyhow::Result<NPResult, (anyhow::Error, NPResult)> {
        self.psi = psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(self.c, self.gamma, &self.error_type),
            false,
            false,
        );
        // let obs_pred = get_population_predictions(
        //     &self.equation,
        //     &self.data,
        //     &self.theta,
        //     false,
        //     self.cycle == 1,
        // );

        // self.psi = obs_pred.get_psi(&ErrorModel::new(self.c, self.gamma, &self.error_type));
        let (w, objf) = burke(&self.psi).expect("Error in IPM");
        self.w = w;
        self.objf = objf;
        Ok(self.to_npresult())
    }
}
