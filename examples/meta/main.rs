#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::prelude::*;

fn main() {
    let eq = ode! {
        diffeq: |x, p, t, dx, b, rateiv, cov| {
            fetch_cov!(cov, t, wt, pkvisit);
            fetch_params!(p, cls, fm, k20, relv, theta1, theta2, vs);
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            let v2 = relv * v;
            dx[0] = rateiv[0] - ke * x[0] * (1.0 - fm) - fm * x[0] + b[0];
            dx[1] = fm * x[0] - k20 * x[1];
        },
        out: |x, p, t, cov, y| {
            fetch_cov!(cov, t, wt, pkvisit);
            fetch_params!(p, cls, fm, k20, relv, theta1, theta2, vs);
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            let v2 = relv * v;
            y[0] = x[0] / v;
            y[1] = x[1] / v2;
        },
    };

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .add_channel(ObservationChannel::continuous(1, "metabolite"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
                )
                .unwrap()
                .add(
                    1,
                    AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("cls", 0.1, 10.0))
                .add(ParameterSpec::bounded("fm", 0.0, 1.0))
                .add(ParameterSpec::bounded("k20", 0.01, 1.0))
                .add(ParameterSpec::bounded("relv", 0.1, 1.0))
                .add(ParameterSpec::bounded("theta1", 0.1, 10.0))
                .add(ParameterSpec::bounded("theta2", 0.1, 10.0))
                .add(ParameterSpec::bounded("vs", 1.0, 10.0)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/meta/meta.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npod(
            NpodOptions::default(),
        )))
        .runtime(RuntimeOptions {
            cycles: 10000,
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
