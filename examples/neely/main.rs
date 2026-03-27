use pmcore::prelude::*;
fn main() {
    let ode = ode! {
        diffeq: |x, p, t, dx, b, rateiv, cov| {
            fetch_params!(p, cls, k30, k40, qs, vps, vs, fm1, fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);

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

            //</tem>
            dx[0] = rateiv[0] - ke * x[0] * (1.0 - fm1 - fm2) - (fm1 + fm2) * x[0] - k12 * x[0]
                + k21 * x[1]
                + b[0];
            dx[1] = k12 * x[0] - k21 * x[1];
            dx[2] = fm1 * x[0] - k30 * x[2];
            dx[3] = fm2 * x[0] - k40 * x[3];
        },
        out: |x, p, t, cov, y| {
            fetch_params!(p, cls, _k30, _k40, qs, vps, vs, _fm1, _fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);

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

            y[0] = x[0] / v;
            y[1] = x[2] / vm1;
            y[2] = x[3] / vm2;
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
