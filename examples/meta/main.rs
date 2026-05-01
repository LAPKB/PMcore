#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::prelude::*;

fn main() {
    let eq = ode! {
        name: "meta",
        params: [cls, fm, k20, relv, theta1, theta2, vs],
        covariates: [wt, pkvisit],
        states: [central, metabolite],
        outputs: [1, 2],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            dx[central] = -ke * x[central] * (1.0 - fm) - fm * x[central];
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

    let data = data::read_pmetrics("examples/meta/meta.csv").unwrap();
    EstimationProblem::builder(eq, data)
        .parameter(Parameter::bounded("cls", 0.1, 10.0))
        .unwrap()
        .parameter(Parameter::bounded("fm", 0.0, 1.0))
        .unwrap()
        .parameter(Parameter::bounded("k20", 0.01, 1.0))
        .unwrap()
        .parameter(Parameter::bounded("relv", 0.1, 1.0))
        .unwrap()
        .parameter(Parameter::bounded("theta1", 0.1, 10.0))
        .unwrap()
        .parameter(Parameter::bounded("theta2", 0.1, 10.0))
        .unwrap()
        .parameter(Parameter::bounded("vs", 1.0, 10.0))
        .unwrap()
        .method(Npod::new())
        .error(
            "1",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .error(
            "2",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .cycles(10000)
        .fit()
        .unwrap();
}
