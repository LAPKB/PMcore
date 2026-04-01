use pmcore::prelude::*;

fn main() {
    let analytical = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] * 1000.0 / v;
        },
    );

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(analytical)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ka", 0.001, 3.0))
                .add(ParameterSpec::bounded("ke", 0.001, 3.0))
                .add(ParameterSpec::bounded("v", 0.001, 50.0)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/theophylline/theophylline.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
