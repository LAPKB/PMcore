use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let sde = sde! {
        name: "iov",
        params: [ke0],
        states: [central, ke_latent],
        outputs: [outeq_1],
        particles: 10000,
        routes: [
            bolus(input_1) -> central,
        ],
        drift: |x, _t, dx| {
            dx[ke_latent] = -x[ke_latent] + ke0;
            dx[central] = -x[ke_latent] * x[central];
        },
        diffusion: |sigma| {
            sigma[ke_latent] = 0.1;
        },
        init: |_t, x| {
            x[ke_latent] = ke0;
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / 50.0;
        },
    };

    let data = data::read_pmetrics("examples/iov/test.csv").unwrap();
    EstimationProblem::builder(sde, data)
        .nonparametric()
        .parameter(Parameter::bounded("ke0", 0.001, 2.0))
        .error_model(
            "outeq_1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.0, 0.0, 0.0), 0.0000757575757576),
        )
        .build()?
        .fit_with(NpagConfig::default())
        .unwrap();

    Ok(())
}

