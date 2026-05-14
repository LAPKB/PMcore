use pmcore::prelude::*;

fn main() {
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

    let data = data::read_pmetrics("examples/theophylline/theophylline.csv").unwrap();
    EstimationProblem::builder(analytical, data)
        .parameter(Parameter::bounded("ka", 0.001, 3.0))
        .unwrap()
        .parameter(Parameter::bounded("ke", 0.001, 3.0))
        .unwrap()
        .parameter(Parameter::bounded("v", 0.001, 50.0))
        .unwrap()
        .method(Npag::new())
        .error(
            "outeq_0",
            AssayErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0),
        )
        .unwrap()
        .fit()
        .unwrap();
}
