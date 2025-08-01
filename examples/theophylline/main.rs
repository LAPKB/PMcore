use pmcore::prelude::*;

fn main() {
    // let eq = Equation::new_ode(
    //     |x, p, _t, dx, rateiv, _cov| {
    //         // fetch_cov!(cov, t, wt);
    //         fetch_params!(p, ka, ke, _v);
    //         dx[0] = -ka * x[0];
    //         dx[1] = ka * x[0] - ke * x[1];
    //     },
    //     |_p, _t, _cov| lag! {},
    //     |_p, _t, _cov| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ka, _ke, v);
    //         y[0] = x[1] * 1000.0 / v;
    //     },
    //     (2, 1),
    // );
    let eq = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] * 1000.0 / v;
        },
        (2, 1),
    );

    let params = Parameters::new()
        .add("ka", 0.001, 3.0)
        .add("ke", 0.001, 3.0)
        .add("v", 0.001, 50.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0, None),
        )
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.initialize_logs().unwrap();
    let data = data::read_pmetrics("examples/theophylline/theophylline.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    // let result = algorithm.fit().unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
