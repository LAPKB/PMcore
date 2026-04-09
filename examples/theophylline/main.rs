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
    )
    .with_default_cache();

    let params = Parameters::new()
        .add("ka", 0.001, 3.0)
        .add("ke", 0.001, 3.0)
        .add("v", 0.001, 50.0);

    let ems = AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0),
        )
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.initialize_logs().unwrap();
    let data = data::read_pmetrics("examples/theophylline/theophylline.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, analytical, data).unwrap();
    // let result = algorithm.fit().unwrap();
    algorithm.initialize().unwrap();
    let mut result = algorithm.fit().unwrap();
    result.write_outputs().unwrap();
}
