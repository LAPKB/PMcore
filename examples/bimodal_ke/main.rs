use anyhow::Result;
use pmcore::{output::logging::Logger, prelude::*};

fn main() -> Result<()> {
    Logger::new().init()?;

    let eq = ode! {
        name: "bimodal_ke",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [
            infusion(input_1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;

    let _result = EstimationProblem::builder(eq, data)
        .algorithm(Npag::default())
        .parameter(Parameter::bounded("ke", 0.001, 3.0))?
        .parameter(Parameter::bounded("v", 25.0, 250.0))?
        .error(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )?
        .fit()?;

    Ok(())
}
