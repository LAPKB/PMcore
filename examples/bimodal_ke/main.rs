use anyhow::Result;
use pmcore::{model::BoundedParameter, prelude::*};
// (Assuming BoundedParameter, NpagConfig, etc., are exported in your prelude)

fn main() -> Result<()> {
    Logger::new().stdout(true).init()?;

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

    let problem = EstimationProblem::builder(eq, data)
        .nonparametric()
        .parameter(BoundedParameter::new("ke", 0.001, 3.0))
        .parameter(BoundedParameter::new("v", 25.0, 250.0))
        .error(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 5.0),
        )
        .build()?;

    let _result = problem.fit_with(NpagConfig::default())?;

    Ok(())
}
