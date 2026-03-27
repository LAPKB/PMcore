use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        diffeq: |x, p, _t, dx, b, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[1] + b[1];
        },
        out: |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[0] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(1, "cp"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    1,
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
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
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
