use pmcore::prelude::*;

fn main() -> Result<()> {
    let sde = sde! {
        name: "new_iov",
        params: [ke0, ske],
        states: [central, ke_latent],
        outputs: [outeq_1],
        particles: 11,
        routes: [
            bolus(input_1) -> central,
        ],
        drift: |x, _t, dx| {
            dx[ke_latent] = -x[ke_latent] + ke0;
            dx[central] = -x[ke_latent] * x[central];
        },
        diffusion: |sigma| {
            sigma[ke_latent] = ske;
        },
        init: |_t, x| {
            x[ke_latent] = ke0;
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / 50.0;
        },
    };

    let data = data::read_pmetrics("examples/new_iov/data.csv")?;
    let parameters = ParameterSpace::bounded()
        .add("ke0", 0.0001, 2.4)
        .add("ske", 0.0001, 0.2);
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537), 0.0),
    )?;
    EstimationProblem::nonparametric(sde, data, prior, error_models)?
        .fit_with(NpagConfig::default())?;

    Ok(())
}

