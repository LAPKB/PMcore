use pmcore::prelude::*;

fn main() {
    let sde = sde! {
        name: "vanco_sde",
        params: [ka, ke0, kcp, kpc, vol, ske],
        covariates: [wt],
        states: [depot, central, peripheral, ke_latent],
        outputs: [outeq_1],
        particles: 100,
        routes: [
            bolus(input_1) -> depot,
        ],
        drift: |x, _t, dx| {
            dx[ke_latent] = -x[ke_latent] + ke0;
            let ke = x[ke_latent];
            dx[depot] = -ka * x[depot];
            dx[central] = ka * x[depot] - (ke + kcp) * x[central] + kpc * x[peripheral];
            dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
        },
        diffusion: |sigma| {
            sigma[ke_latent] = ske;
        },
        init: |_t, x| {
            x[ke_latent] = ke0;
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / (vol * wt);
        },
    };

    // let ode = equation::ODE::new(
    //     |x, p, _t, dx, _rateiv, _cov| {
    //         fetch_params!(p, ka, ke0, kcp, kpc, _vol);
    //         dx[0] = -ka * x[0];
    //         dx[1] = ka * x[0] - (ke0 + kcp) * x[1] + kpc * x[2];
    //         dx[2] = kcp * x[1] - kpc * x[2];
    //     },
    //     |_p, _t, _cov| lag! {},
    //     |_p, _t, _cov| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, t, cov, y| {
    //         fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol);
    //         fetch_cov!(cov, t, wt);
    //         y[0] = x[1] / (vol);
    //     },
    //     (3, 1),
    // );

    let data = data::read_pmetrics("examples/vanco_sde/vanco_clean.csv").unwrap();
    EstimationProblem::builder(sde, data)
        .parameter(Parameter::bounded("ka", 0.0001, 2.4))
        .unwrap()
        .parameter(Parameter::bounded("ke0", 0.0001, 2.7))
        .unwrap()
        .parameter(Parameter::bounded("kcp", 0.0001, 2.4))
        .unwrap()
        .parameter(Parameter::bounded("kpc", 0.0001, 2.4))
        .unwrap()
        .parameter(Parameter::bounded("vol", 0.2, 12.0))
        .unwrap()
        .parameter(Parameter::bounded("ske", 0.0001, 0.2))
        .unwrap()
        .method(Npag::new())
        .error(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.00119, 0.20, 0.0, 0.0), 0.0),
        )
        .unwrap()
        .output_dir("examples/vanco_sde/output")
        .cycles(usize::MAX)
        .prior(Prior::sobol(100, 347))
        .fit()
        .unwrap();
}
