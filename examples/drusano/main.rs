use anyhow::Result;
use pmcore::prelude::*;
#[allow(unused_variables)]
fn main() -> Result<()> {
    let eq = ode! {
        name: "drusano",
        params: [
            v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1, e50_1r1,
            alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1, h2r2
        ],
        covariates: [ic_t],
        states: [drug_1_amount, drug_2_amount, total_bacteria, resistant_1, resistant_2],
        outputs: [1, 2, 3, 4],
        routes: [
            bolus(1) -> drug_1_amount,
            bolus(2) -> drug_2_amount,
        ],
        diffeq: |x, _t, dx| {

            let e50_2r1 = e50_2s;
            let e50_1r2 = e50_1s;
            let h2r1 = h2s;
            let h1r2 = h1s;

            dx[drug_1_amount] = -cl1 * x[drug_1_amount] / v1;
            dx[drug_2_amount] = -cl2 * x[drug_2_amount] / v2;

            let xns = x[total_bacteria];
            let xnr1 = x[resistant_1];
            let xnr2 = x[resistant_2];
            let e = 1.0 - (xns + xnr1 + xnr2) / popmax;

            let u_s = x[drug_1_amount] / (v1 * e50_1s);
            let v_s = x[drug_2_amount] / (v2 * e50_2s);
            let w_s = alpha_s * u_s * v_s / (e50_1s * e50_2s);
            let xm0best = get_e2(u_s, v_s, w_s, 1.0 / h1s, 1.0 / h2s, alpha_s);
            dx[total_bacteria] = xns * (kgs * e - kks * xm0best);

            let u_r1 = x[drug_1_amount] / (v1 * e50_1r1);
            let v_r1 = x[drug_2_amount] / (v2 * e50_2r1);
            let w_r1 = alpha_r1 * u_r1 * v_r1 / (e50_1r1 * e50_2r1);
            let xm0best = get_e2(u_r1, v_r1, w_r1, 1.0 / h1r1, 1.0 / h2r1, alpha_s);
            dx[resistant_1] = xnr1 * (kgr1 * e - kkr1 * xm0best);

            let u_r2 = x[drug_1_amount] / (v1 * e50_1r2);
            let v_r2 = x[drug_2_amount] / (v2 * e50_2r2);
            let w_r2 = alpha_r2 * u_r2 * v_r2 / (e50_1r2 * e50_2r2);
            let xm0best = get_e2(u_r2, v_r2, w_r2, 1.0 / h1r2, 1.0 / h2r2, alpha_s);
            dx[resistant_2] = xnr2 * (kgr2 * e - kkr2 * xm0best);
        },
        init: |_t, x| {
            x[drug_1_amount] = 0.0;
            x[drug_2_amount] = 0.0;
            x[total_bacteria] = 10.0_f64.powf(ic_t);
            x[resistant_1] = 10.0_f64.powf(init_4);
            x[resistant_2] = 10.0_f64.powf(init_5);
        },
        out: |x, _t, y| {
            y[1] = x[drug_1_amount] / v1;
            y[2] = x[drug_2_amount] / v2;
            y[3] = (x[total_bacteria] + x[resistant_1] + x[resistant_2]).log10();
            y[4] = x[resistant_1].log10();
        },
    };

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "drug_1"))
        .add_channel(ObservationChannel::continuous(1, "drug_2"))
        .add_channel(ObservationChannel::continuous(2, "total"))
        .add_channel(ObservationChannel::continuous(3, "resistant_1"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0),
                )?
                .add(
                    1,
                    AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0),
                )?
                .add(
                    2,
                    AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0),
                )?
                .add(
                    3,
                    AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0),
                )?,
        );

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("v1", 5.0, 160.0))
                .add(ParameterSpec::bounded("cl1", 4.0, 9.0))
                .add(ParameterSpec::bounded("v2", 100.0, 200.0))
                .add(ParameterSpec::bounded("cl2", 25.0, 35.0))
                .add(ParameterSpec::bounded(
                    "popmax",
                    100000000.0,
                    100000000000.0,
                ))
                .add(ParameterSpec::bounded("kgs", 0.01, 0.25))
                .add(ParameterSpec::bounded("kks", 0.01, 0.5))
                .add(ParameterSpec::bounded("e50_1s", 0.1, 2.5))
                .add(ParameterSpec::bounded("e50_2s", 0.1, 10.0))
                .add(ParameterSpec::bounded("alpha_s", -8.0, 5.0))
                .add(ParameterSpec::bounded("kgr1", 0.004, 0.1))
                .add(ParameterSpec::bounded("kkr1", 0.08, 0.4))
                .add(ParameterSpec::bounded("e50_1r1", 8.0, 17.0))
                .add(ParameterSpec::bounded("alpha_r1", -8.0, 5.0))
                .add(ParameterSpec::bounded("kgr2", 0.004, 0.3))
                .add(ParameterSpec::bounded("kkr2", 0.1, 0.5))
                .add(ParameterSpec::bounded("e50_2r2", 5.0, 8.0))
                .add(ParameterSpec::bounded("alpha_r2", -5.0, 5.0))
                .add(ParameterSpec::bounded("init_4", -1.0, 4.0))
                .add(ParameterSpec::bounded("init_5", -1.0, 3.0))
                .add(ParameterSpec::bounded("h1s", 0.5, 8.0))
                .add(ParameterSpec::bounded("h2s", 0.1, 4.0))
                .add(ParameterSpec::bounded("h1r1", 5.0, 25.0))
                .add(ParameterSpec::bounded("h2r2", 10.0, 22.0)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/drusano/data.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan {
            write: true,
            path: Some("examples/drusano/output".to_string()),
        })
        .runtime(RuntimeOptions {
            prior: Some(Prior::sobol(212900, 347)),
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();
    Ok(())
}
