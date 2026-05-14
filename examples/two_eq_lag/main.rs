use pmcore::prelude::*;

fn main() {
    let eq = ode! {
        name: "two_eq_lag",
        params: [ka, ke, tlag, v],
        states: [gut, central],
        outputs: [outeq_0],
        routes: [
            bolus(input_0) -> gut,
        ],
        diffeq: |x, _t, dx| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        lag: |_t| {
            lag! { input_0 => tlag }
        },
        out: |x, _t, y| {
            y[outeq_0] = x[central] / v;
        },
    };

    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();
    EstimationProblem::builder(eq, data)
        .parameter(Parameter::bounded("ka", 0.1, 0.9))
        .unwrap()
        .parameter(Parameter::bounded("ke", 0.001, 0.1))
        .unwrap()
        .parameter(Parameter::bounded("tlag", 0.0, 4.0))
        .unwrap()
        .parameter(Parameter::bounded("v", 30.0, 120.0))
        .unwrap()
        .method(Npag::new())
        .error(
            "outeq_0",
            AssayErrorModel::additive(ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537), 0.0),
        )
        .unwrap()
        .output_dir("examples/two_eq_lag/output")
        .initialize_logs()
        .fit()
        .unwrap();
}
