use pmcore::prelude::*;
fn main() {
    let ode = ode! {
        name: "neely",
        params: [cls, k30, k40, qs, vps, vs, fm1, fm2, theta1, theta2],
        covariates: [wt, pkvisit],
        states: [central, peripheral, metabolite_1, metabolite_2],
        outputs: [1, 2, 3],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            let vfrac1 = 0.068202;
            let vfrac2 = 0.022569;
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let ke = cl / v;
            let _vm1 = vfrac1 * v;
            let _vm2 = vfrac2 * v;
            let k12 = q / v;
            let k21 = q / vp;

            dx[central] = -ke * x[central] * (1.0 - fm1 - fm2)
                - (fm1 + fm2) * x[central]
                - k12 * x[central]
                + k21 * x[peripheral];
            dx[peripheral] = k12 * x[central] - k21 * x[peripheral];
            dx[metabolite_1] = fm1 * x[central] - k30 * x[metabolite_1];
            dx[metabolite_2] = fm2 * x[central] - k40 * x[metabolite_2];
        },
        out: |x, _t, y| {
            let vfrac1 = 0.068202;
            let vfrac2 = 0.022569;
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let _ke = cl / v;
            let vm1 = vfrac1 * v;
            let vm2 = vfrac2 * v;
            let _k12 = q / v;
            let _k21 = q / vp;

            y[1] = x[central] / v;
            y[2] = x[metabolite_1] / vm1;
            y[3] = x[metabolite_2] / vm2;
        },
    };
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .add_channel(ObservationChannel::continuous(1, "m1"))
        .add_channel(ObservationChannel::continuous(2, "m2"))
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
                .unwrap()
                .add(
                    2,
                    AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(ode)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("cls", 0.0, 0.4))
                .add(ParameterSpec::bounded("k30", 0.0, 0.5))
                .add(ParameterSpec::bounded("k40", 0.3, 1.5))
                .add(ParameterSpec::bounded("qs", 0.0, 0.5))
                .add(ParameterSpec::bounded("vps", 0.0, 5.0))
                .add(ParameterSpec::bounded("vs", 0.0, 2.0))
                .add(ParameterSpec::bounded("fm1", 0.0, 0.2))
                .add(ParameterSpec::bounded("fm2", 0.0, 0.1))
                .add(ParameterSpec::bounded("theta1", -4.0, 2.0))
                .add(ParameterSpec::bounded("theta2", -2.0, 0.5)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/neely/data.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npsah(
            NpsahOptions::default(),
        )))
        .output(OutputPlan {
            write: true,
            path: Some("examples/neely/output/".to_string()),
        })
        .runtime(RuntimeOptions {
            cycles: 1000,
            prior: Some(Prior::sobol(2028, 22)),
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
