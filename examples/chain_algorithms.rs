//! Algorithm chaining: NPAG(10) → NPOD(5) on a simple 1-compartment IV bolus model.
//!
//! Demonstrates `NonParametricResult::chain()`.
//! Uses inline synthetic data to keep the example self-contained.

use pharmsol::SubjectBuilderExt;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        name: "chain_example",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [bolus(input_1) -> central],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    };

    let subject1 = Subject::builder("1")
        .bolus(0.0, 100.0, 1)
        .observation(1.0, 8.0, 1)
        .observation(2.0, 5.0, 1)
        .observation(4.0, 2.0, 1)
        .build();

    let subject2 = Subject::builder("2")
        .bolus(0.0, 80.0, 1)
        .observation(1.0, 6.0, 1)
        .observation(2.0, 4.0, 1)
        .observation(4.0, 1.5, 1)
        .build();

    let data = Data::new(vec![subject1, subject2]);

    let parameters = ParameterSpace::bounded()
        .add("ke", 0.001, 3.0)
        .add("v", 5.0, 50.0);
    let prior = Theta::sobol(&parameters, 100)?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?;

    let result = EstimationProblem::nonparametric(eq, data, prior, error_models)?
        .fit_with(NpagConfig::new().max_cycles(10))?
        .chain(NpodConfig::new().max_cycles(5))?;

    println!(
        "Chained NPAG→NPOD: OBJF = {:.2}, {} support points, {} total cycles",
        result.objf(),
        result.get_theta().nspp(),
        result.cycles(),
    );

    Ok(())
}
