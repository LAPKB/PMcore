use pharmsol::{prelude::simulator::SubjectPredictions, Equation, Subject, ODE};

/// This simulator defines the simulator in a high level abstraction.
/// Understand it as the simulator to be used by final users. Not the simulator used internally by the algorithms.

pub fn simulate(subject: &Subject, eqn: &ODE, support_point: &Vec<f64>) -> SubjectPredictions {
    eqn.estimate_predictions(subject, support_point)
}
