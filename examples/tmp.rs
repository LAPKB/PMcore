use pmcore::prelude::{data::read_pmetrics, *};

fn main() {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let data = read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let spp = support_point!("ke" => 1.5, "v" => 125.0);
    let theta = data.estimate_theta(&eq, &spp);
    dbg!(theta);
}
