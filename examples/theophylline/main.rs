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
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("theophylline")
            .parameters(["ka", "ke", "v"])
            .states(["depot", "central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("depot"))
            .analytical_kernel(AnalyticalKernel::OneCompartmentWithAbsorption),
    )
    .unwrap();

    let data = data::read_pmetrics("examples/theophylline/theophylline.csv").unwrap();
    EstimationProblem::builder(analytical, data)
        .parameter(Parameter::bounded("ka", 0.001, 3.0))
        .unwrap()
        .parameter(Parameter::bounded("ke", 0.001, 3.0))
        .unwrap()
        .parameter(Parameter::bounded("v", 0.001, 50.0))
        .unwrap()
        .method(Npag::new())
        .error(
            "0",
            AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0),
        )
        .unwrap()
        .fit()
        .unwrap();
}
