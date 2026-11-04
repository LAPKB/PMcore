use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    Logger::new().stdout(true).init()?;

    let eq = ode! {
        name: "bimodal_ke",
        params: [ke, v],
        states: [central],
        outputs: [1],
        diffeq: |x, _t, dx| {
            dx[central] = infusion[1] - ke * x[central];
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;

    let parameters = ParameterSpace::bounded()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let prior = Theta::sobol_default(&parameters)?;

    let error_models = AssayErrorModels::new().add(
        "1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?;

    let problem = EstimationProblem::nonparametric(eq, data, prior, error_models)?;

    let _result = problem.fit_with(NpagConfig::new())?;

    _result.write_outputs("outputs/bimodal_ke", PredictionGrid::new())?;

    Ok(())
}
