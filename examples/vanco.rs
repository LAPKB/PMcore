use pmcore::prelude::{simulator::Equation, *};
fn main() {
    let eq = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ke, kcp, kpc);
            dx[0] = -ke * x[0] - kcp * x[0] + kpc * x[1];
            dx[1] = -kpc * x[1] + kcp * x[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 500.0;
        },
        |x, _p, _t, _cov, y| {
            y[0] = x[1];
        },
        (2, 1),
    );
    // same eq but analytical
    // let eq = Equation::new_analytical(
    //     two_compartments,
    //     |_p, _cov| {},
    //     |_p| lag! {},
    //     |_p| fa! {},
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

    let op = eq.simulate_subject(&subject, &vec![0.3, 0.2, 0.5], None).0;

    let times = op.flat_time();
    let pred = op.flat_predictions();

    for (t, p) in times.iter().zip(pred.iter()) {
        println!("{}, {}", t, p);
    }
}
