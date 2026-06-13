use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        name: "neely",
        params: [cls, k30, k40, qs, vps, vs, fm1, fm2, theta1, theta2],
        covariates: [wt, pkvisit],
        states: [central, peripheral, metabolite_1, metabolite_2],
        outputs: [outeq_1, outeq_2, outeq_3],
        routes: [
            infusion(input_1) -> central,
        ],
        diffeq: |x, _t, dx| {
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let ke = cl / v;
            let k12 = q / v;
            let k21 = q / vp;

            dx[central] = -ke * x[central] * (1.0 - fm1 - fm2)
                - (fm1 + fm2) * x[central]
                - k12 * x[central]
                + k21 * x[peripheral];
            dx[peripheral] = k12 * x[central] - k21 * x[peripheral];
            dx[metabolite_1] = fm1 * x[central] - k30 * x[metabolite_1];
            dx[metabolite_2] = fm2 * x[central] - k40 * x[metabolite_2];
        },
        out: |x, _t, y| {
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

            y[outeq_1] = x[central] / v;
            y[outeq_2] = x[metabolite_1] / vm1;
            y[outeq_3] = x[metabolite_2] / vm2;
        },
    };

    let data = data::read_pmetrics("examples/neely/data.csv")?;
    let parameters = ParameterSpace::bounded()
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
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new()
        .add(
            "outeq_1",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )?
        .add(
            "outeq_2",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )?
        .add(
            "outeq_3",
            AssayErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )?;
    EstimationProblem::nonparametric(eq, data, prior, error_models)?
        .fit_with(NonParametricAlgorithm::npag())?;

    Ok(())
}

