use crate::{
    prelude::{
        algorithms::Algorithm,
        datafile::Scenario,
        evaluation::sigma::{ErrorPoly, ErrorType},
        ipm_faer::burke,
        output::NPResult,
        prob,
        settings::Settings,
    },
    simulator::Equation,
    tui::ui::Comm,
};

use ndarray::{Array1, Array2};
use tokio::sync::mpsc::UnboundedSender;

use super::{data::Subject, get_obspred};

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct POSTPROB {
    equation: Equation,
    psi: Array2<f64>,
    theta: Array2<f64>,
    w: Array1<f64>,
    objf: f64,
    cycle: usize,
    converged: bool,
    gamma: f64,
    error_type: ErrorType,
    subjects: Vec<Subject>,
    c: (f64, f64, f64, f64),
    #[allow(dead_code)]
    tx: Option<UnboundedSender<Comm>>,
    settings: Settings,
}

impl Algorithm for POSTPROB {
    fn fit(&mut self) -> NPResult {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
            self.subjects.clone(),
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

impl POSTPROB {
    pub fn new(
        equation: Equation,
        theta: Array2<f64>,
        subjects: Vec<Subject>,
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
            subjects,
            c,
        }
    }

    pub fn run(&mut self) -> NPResult {
        let obs_pred = get_obspred(&self.equation, &self.subjects, &self.theta, false);

        self.psi = obs_pred.likelihood(&ErrorPoly {
            c: self.c,
            gl: self.gamma,
            e_type: &self.error_type,
        });
        let (w, objf) = burke(&self.psi).expect("Error in IPM");
        self.w = w;
        self.objf = objf;
        self.to_npresult()
    }
}
