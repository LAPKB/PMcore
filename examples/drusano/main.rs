use anyhow::Result;
use pmcore::prelude::*;
#[allow(unused_variables)]
fn main() -> Result<()> {
    let eq = equation::ODE::new(
        |x, p, t, dx, rateiv, _cov| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1,
                h2r2
            );

            let e50_2r1 = e50_2s;
            let e50_1r2 = e50_1s;
            let h2r1 = h2s;
            let h1r2 = h1s;

            dx[0] = rateiv[0] - cl1 * x[0] / v1;
            dx[1] = rateiv[1] - cl2 * x[1] / v2;

            let xns = x[2];
            let xnr1 = x[3];
            let xnr2 = x[4];
            let e = 1.0 - (xns + xnr1 + xnr2) / popmax;

            // Case s
            let u_s = x[0] / (v1 * e50_1s);
            let v_s = x[1] / (v2 * e50_2s);
            let w_s = alpha_s * u_s * v_s / (e50_1s * e50_2s);
            let xm0best = get_xm0best(u_s, v_s, w_s, 1.0 / h1s, 1.0 / h2s, alpha_s);
            dx[2] = xns * (kgs * e - kks * xm0best);

            // Case r1
            let u_r1 = x[0] / (v1 * e50_1r1);
            let v_r1 = x[1] / (v2 * e50_2r1);
            let w_r1 = alpha_r1 * u_r1 * v_r1 / (e50_1r1 * e50_2r1);
            let xm0best = get_xm0best(u_r1, v_r1, w_r1, 1.0 / h1r1, 1.0 / h2r1, alpha_s);
            dx[3] = xnr1 * (kgr1 * e - kkr1 * xm0best);

            // Case r2
            let u_r2 = x[0] / (v1 * e50_1r2);
            let v_r2 = x[1] / (v2 * e50_2r2);
            let w_r2 = alpha_r2 * u_r2 * v_r2 / (e50_1r2 * e50_2r2);
            let xm0best = get_xm0best(u_r2, v_r2, w_r2, 1.0 / h1r2, 1.0 / h2r2, alpha_s);
            dx[4] = xnr2 * (kgr2 * e - kkr2 * xm0best);
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, t, cov, x| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1,
                h2r2
            );
            fetch_cov!(cov, t, ic_t);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 10.0_f64.powf(ic_t);
            x[3] = 10.0_f64.powf(init_4);
            x[4] = 10.0_f64.powf(init_5);
        },
        |x, p, _t, _cov, y| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1,
                h2r2
            );
            y[0] = x[0] / v1;
            y[1] = x[1] / v2;
            y[2] = (x[2] + x[3] + x[4]).log10();
            y[3] = x[3].log10();
            y[4] = x[4].log10();
        },
        (5, 5),
    );

    let params = Parameters::new()
        .add("v1", 5.0, 160.0)
        .add("cl1", 4.0, 9.0)
        .add("v2", 100.0, 200.0)
        .add("cl2", 25.0, 35.0)
        .add("popmax", 100000000.0, 100000000000.0)
        .add("kgs", 0.01, 0.25)
        .add("kks", 0.01, 0.5)
        .add("e50_1s", 0.1, 2.5)
        .add("e50_2s", 0.1, 10.0)
        .add("alpha_s", -8.0, 5.0)
        .add("kgr1", 0.004, 0.1)
        .add("kkr1", 0.08, 0.4)
        .add("e50_1r1", 8.0, 17.0)
        .add("alpha_r1", -8.0, 5.0)
        .add("kgr2", 0.004, 0.3)
        .add("kkr2", 0.1, 0.5)
        .add("e50_2r2", 5.0, 8.0)
        .add("alpha_r2", -5.0, 5.0)
        .add("init_4", -1.0, 4.0)
        .add("init_5", -1.0, 3.0)
        .add("h1s", 0.5, 8.0)
        .add("h2s", 0.1, 4.0)
        .add("h1r1", 5.0, 25.0)
        .add("h2r2", 10.0, 22.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0, None),
        )?
        .add(
            1,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0, None),
        )?
        .add(
            2,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0, None),
        )?
        .add(
            3,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0, None),
        )?
        .add(
            4,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 1.0, None),
        )?;

    let mut settings = SettingsBuilder::new()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_prior(Prior::sobol(212900, 347));
    settings.set_output_path("examples/drusano/output");

    settings.initialize_logs()?;
    let data = data::read_pmetrics("examples/drusano/data.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    algorithm.initialize().unwrap();
    algorithm.fit().unwrap();
    // while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
    Ok(())
}
