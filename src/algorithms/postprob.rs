use crate::{
    prelude::{algorithms::Algorithm, ipm::burke, output::NPResult, settings::Settings},
    tui::ui::Comm,
};
use anyhow::{Context, Error};
use pharmsol::prelude::{
    data::{Data, ErrorModel, ErrorType},
    simulator::{psi, Equation},
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

impl<E: Equation> Algorithm<E> for POSTPROB<E> {
    type Matrix = Array2<f64>;

    fn new(config: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        let theta = Array2::zeros((data.get_subjects().len(), 1));
        let c = (0.0, 0.0, 0.0, 0.0);

        Ok(Box::new(Self {
            equation,
            psi: Array2::default((0, 0)),
            theta,
            w: Array1::default(0),
            objf: f64::INFINITY,
            cycle: 0,
            converged: false,
            gamma: config.error.value,
            error_type: match config.error.class.as_str() {
                "additive" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            tx: None,
            settings: config,
            data,
            c,
            cyclelog: CycleLog::new(),
        }))
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
    fn get_settings(&self) -> &Settings {
        &self.settings
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Self::Matrix {
        unimplemented!()
    }

    fn set_theta(&mut self, theta: Self::Matrix) {
        self.theta = theta;
    }

    fn converge_criteria(&self) -> bool {
        unimplemented!()
    }

    fn evaluation(&mut self) -> Result<(), (Error, NPResult)> {
        unimplemented!()
    }

    fn filter(&mut self) -> Result<(), (Error, NPResult)> {
        unimplemented!()
    }

    fn expansion(&mut self) -> Result<(), (Error, NPResult)> {
        unimplemented!()
    }
}

// impl<E: Equation> POSTPROB<E> {
//     pub fn run(self) -> anyhow::Result<NPResult, (anyhow::Error, NPResult)> {
//         self.psi = psi(
//             &self.equation,
//             &self.data,
//             &self.theta,
//             &ErrorModel::new(self.c, self.gamma, &self.error_type),
//             false,
//             false,
//         );
//         // let obs_pred = get_population_predictions(
//         //     &self.equation,
//         //     &self.data,
//         //     &self.theta,
//         //     false,
//         //     self.cycle == 1,
//         // );

//         // self.psi = obs_pred.get_psi(&ErrorModel::new(self.c, self.gamma, &self.error_type));
//         let (w, objf) = burke(&self.psi).expect("Error in IPM");
//         self.w = w;
//         self.objf = objf;
//         Ok(self.to_npresult())
//     }
// }
