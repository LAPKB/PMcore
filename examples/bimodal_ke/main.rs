use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        name: "bimodal_ke",
        params: [ke, v],
        states: [central],
        outputs: [1],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    let _result = EstimationProblem::builder(eq, data)
        .method(Npag::default())
        .parameter(Parameter::bounded("ke", 0.001, 3.0))?
        .parameter(Parameter::bounded("v", 25.0, 250.0))?
        .error(
            "1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )?
        .output_dir("examples/bimodal_ke/output/")
        .cycles(1000)
        .initialize_logs()
        .fit()?;

    Ok(())
}
