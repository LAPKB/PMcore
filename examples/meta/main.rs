#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        name: "meta",
        params: [cls, fm, k20, relv, theta1, theta2, vs],
        covariates: [wt, pkvisit],
        states: [central, metabolite],
        outputs: [1, 2],
        diffeq: |x, _t, dx| {
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            dx[central] = infusion[1] - ke * x[central] * (1.0 - fm) - fm * x[central];
            dx[metabolite] = fm * x[central] - k20 * x[metabolite];
        },
        out: |x, _t, y| {
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let v2 = relv * v;
            let _ke = cl / v;
            y[1] = x[central] / v;
            y[2] = x[metabolite] / v2;
        },
    };

    let data = data::read_pmetrics("examples/meta/meta.csv")?;
    let parameters = ParameterSpace::bounded()
        .add("cls", 0.1, 10.0)
        .add("fm", 0.0, 1.0)
        .add("k20", 0.01, 1.0)
        .add("relv", 0.1, 1.0)
        .add("theta1", 0.1, 10.0)
        .add("theta2", 0.1, 10.0)
        .add("vs", 1.0, 10.0);
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new()
        .add(
            "1",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )?
        .add(
            "2",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )?;
    EstimationProblem::nonparametric(eq, data, prior, error_models)?
        .fit_with(NonParametricAlgorithm::npod())?;

    Ok(())
}
