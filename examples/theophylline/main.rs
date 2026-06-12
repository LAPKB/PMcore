use pmcore::prelude::*;

fn main() -> Result<()> {
    let analytical = analytical! {
        name: "theophylline",
        params: [ka, ke, v],
        states: [depot, central],
        outputs: [outeq_0],
        routes: [
            bolus(input_0) -> depot,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _t, y| {
            y[outeq_0] = x[central] * 1000.0 / v;
        },
    };

    let data = data::read_pmetrics("examples/theophylline/theophylline.csv")?;
    let parameters = ParameterSpace::bounded()
        .add("ka", 0.001, 3.0)
        .add("ke", 0.001, 3.0)
        .add("v", 0.001, 50.0);
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new().add(
        "outeq_0",
        AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0),
    )?;
    EstimationProblem::nonparametric(analytical, data, prior, error_models)?
        .fit_with(NpagConfig::default())?;

    Ok(())
}

