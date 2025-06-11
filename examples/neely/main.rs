use pmcore::prelude::*;
fn main() {
    let ode = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
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
                + k21 * x[1];
            dx[1] = k12 * x[0] - k21 * x[1];
            dx[2] = fm1 * x[0] - k30 * x[2];
            dx[3] = fm2 * x[0] - k40 * x[3];
        },
        |_p| {
            lag! {}
        },
        |_p| {
            fa! {}
        },
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
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
        (4, 3),
    );
    let params = Parameters::new()
        .add("cls", 0.0, 0.4)
        .add("k30", 0.0, 0.5)
        .add("k40", 0.3, 1.5)
        .add("qs", 0.0, 0.5)
        .add("vps", 0.0, 5.0)
        .add("vs", 0.0, 2.0)
        .add("fm1", 0.0, 0.2)
        .add("fm2", 0.0, 0.1)
        .add("theta1", -4.0, 2.0)
        .add("theta2", -2.0, 0.5);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .add(
            1,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .add(
            2,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap();
    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(1000);
    settings.set_prior(Prior::sobol(2028, 22));
    settings.set_output_path("examples/neely/output/");
    settings.set_write_logs(true);
    settings.write().unwrap();
    settings.initialize_logs().unwrap();
    let data = data::read_pmetrics("examples/neely/data.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, ode, data).unwrap();
    let result = algorithm.fit().unwrap();
    result.write_outputs().unwrap();
}
