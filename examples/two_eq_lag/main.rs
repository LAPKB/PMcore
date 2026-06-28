use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        name: "two_eq_lag",
        params: [ka, ke, tlag, v],
        states: [gut, central],
        outputs: [outeq_0],
        routes: [
            bolus(input_0) -> gut,
        ],
        diffeq: |x, _t, dx| {
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - ke * x[central];
        },
        lag: |_t| {
            lag! { input_0 => tlag }
        },
        out: |x, _t, y| {
            y[outeq_0] = x[central] / v;
        },
    };

    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv")?;
    let parameters = ParameterSpace::bounded()
        .add("ka", 0.1, 0.9)
        .add("ke", 0.001, 0.1)
        .add("tlag", 0.0, 4.0)
        .add("v", 30.0, 120.0);
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new().add(
        "outeq_0",
        AssayErrorModel::additive(ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537), 0.0),
    )?;
    EstimationProblem::nonparametric(eq, data, prior, error_models)?
        .fit_with(NonParametricAlgorithm::npag())?;

    Ok(())
}
