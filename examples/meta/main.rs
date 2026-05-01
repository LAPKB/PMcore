#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::prelude::*;

fn main() {
    let eq = ode! {
        name: "meta",
        params: [cls, fm, k20, relv, theta1, theta2, vs],
        covariates: [wt, pkvisit],
        states: [central, metabolite],
        outputs: [1, 2],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            dx[central] = -ke * x[central] * (1.0 - fm) - fm * x[central];
            dx[metabolite] = fm * x[central] - k20 * x[metabolite];
        },
        out: |x, _t, y| {
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let v2 = relv * v;
            let _ke = cl / v;
            y[1] = x[central] / v;
            y[2] = x[metabolite] / v2;
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
            NpodOptions,
        )))
        .runtime(RuntimeOptions {
            cycles: 10000,
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
