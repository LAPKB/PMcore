use pmcore::{logger::setup_log, prelude::*};
use std::io::Write;
fn main() {
    //% cp ~/Documents/CHLA/IOV2024/vanco_PICU.csv ./test.csv
    //% cargo run --release --example vanco_sde

    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke0, kcp, kpc, _vol, _ske);
            dx[3] = -x[3] + ke0;
            let ke = x[3]; // ke is a mean-reverting stochastic parameter
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - (ke + kcp) * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        |p, d| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, _vol, ske);
            d[3] = ske;
        },
        |_p| lag! {}, // remember println!() .. it's a macro ... {} empty function
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ka, ke0, _kcp, _kpc, _vol, _ske);
            //    x[0] = 0.0; // spot check suggests all subjects have a 0-dose at time 0
            //    x[1] = 0.0;
            x[3] = ke0;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol, _ske);
            fetch_cov!(cov, t, wt, ht, male);
            y[0] = x[1] / (vol * wt);
        },
        (4, 1), // (input equations, output equations)
        11,
    );

    let settings = settings::read("examples/w_vanco_sde/config.toml".to_string()).unwrap();
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/w_vanco_sde/test.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
