use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use pharmsol::prelude::*;
use pmcore::prelude::*;
use tracing::Level;
use tracing_subscriber::fmt::MakeWriter;

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer lock")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogBuffer {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn short_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "saem_tracing_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };
    let data = Data::new(vec![Subject::builder("s1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.0, "cp")
        .build()]);

    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.3).fixed())
        .parameter(Parameter::log("v").with_initial(20.0).fixed())
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("tracing fixture should build")
}

#[test]
fn fixed_schedule_fit_emits_honest_lifecycle_logs() {
    let writer = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_max_level(Level::DEBUG)
        .with_writer(writer.clone())
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        short_problem()
            .fit_with(
                SaemConfig::new()
                    .seed(7)
                    .n_chains(1)
                    .mcmc_iterations(1)
                    .burn_in(0)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .compute_map(false),
            )
            .expect("short SAEM fit should complete");
    });

    let bytes = writer.0.lock().expect("log buffer lock").clone();
    let logs = String::from_utf8(bytes).expect("tracing output should be UTF-8");
    assert!(logs.contains("Starting SAEM fit"), "{logs}");
    assert!(logs.contains("Cycle 1"), "{logs}");
    assert!(logs.contains("Conditional N2LL ="), "{logs}");
    assert!(
        logs.contains("Maximum SAEM cycles reached; this is not statistical convergence"),
        "{logs}"
    );
    assert!(!logs.contains("Objective function ="), "{logs}");
}
