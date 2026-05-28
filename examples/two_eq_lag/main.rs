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
        .nonparametric()
        .parameter(BoundedParameter::new("ka", 0.1, 0.9))
        .parameter(BoundedParameter::new("ke", 0.001, 0.1))
        .parameter(BoundedParameter::new("tlag", 0.0, 4.0))
        .parameter(BoundedParameter::new("v", 30.0, 120.0))
        .error(
            "outeq_0",
            AssayErrorModel::additive(ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537), 0.0),
        )
        .build()
        .unwrap()
        .fit_with(NpagConfig::default())
        .unwrap();
}
