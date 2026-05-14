use pmcore::prelude::*;
fn main() {
    let eq = ode! {
        name: "vanco_two_compartment",
        params: [ke, kcp, kpc],
        states: [central, peripheral],
        outputs: [peripheral_amount],
        routes: [
            bolus(dose) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central] - kcp * x[central] + kpc * x[peripheral];
            dx[peripheral] = -kpc * x[peripheral] + kcp * x[central];
        },
        init: |_t, x| {
            x[central] = 500.0;
        },
        out: |x, _t, y| {
            y[peripheral_amount] = x[peripheral];
        },
    };
    // same eq but analytical
    // let eq = Equation::new_analytical(
    //     two_compartments,
    //     |_p, _cov| {},
    //     |_p, _t, _cov| lag! {},
    //     |_p, _t, _cov| fa! {},
    //     |_p, _t, _cov, x| {
    //         x[0] = 500.0;
    //         x[1] = 0.0;
    //     },
    //     |x, _p, _t, _cov, y| {
    //         y[0] = x[0];
    //     },
    //     (2, 1),
    // );

    let subject = data::Subject::builder("id1")
        .observation(0.0, -99.0, 0)
        .repeat(1000, 0.01)
        .build();

    let op = eq
        .simulate_subject_dense(&subject, &[0.3, 0.2, 0.5], None)
        .unwrap()
        .0;

    let times = op.flat_times();
    let pred = op.flat_predictions();

    for (t, p) in times.iter().zip(pred.iter()) {
        println!("{}, {}", t, p);
    }
}
