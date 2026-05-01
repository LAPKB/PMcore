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
    };
    // .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.001, 3.0))
                .add(ParameterSpec::bounded("v", 25.0, 250.0)),
        )
        .observations(observations)
        .build()?;

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(NpagOptions)))
        .output(OutputPlan {
            write: true,
            path: Some("examples/bimodal_ke/output/".to_string()),
        })
        .runtime(RuntimeOptions {
            cycles: 1000,
            logging: LoggingOptions {
                initialize: true,
                ..LoggingOptions::default()
            },
            ..RuntimeOptions::default()
        })
        .run()?;
    result.write_outputs()?;

    Ok(())
}
