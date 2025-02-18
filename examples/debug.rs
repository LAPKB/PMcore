use algorithms::{npag::NPAG, Algorithms};
use ipm::burke;
use logger::setup_log;
use pmcore::{
    prelude::*,
    routines::{evaluation::ipm, logger, settings},
};
fn main() {
    let ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
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
    let an = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let settings = settings::read("examples/bimodal_ke/config.toml").unwrap();
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let mut alg_ode = NPAG::new(settings.clone(), ode, data.clone()).unwrap();
    let mut alg_an = NPAG::new(settings.clone(), an, data.clone()).unwrap();

    // Initialize algorithms
    alg_ode.initialize().unwrap();
    alg_an.initialize().unwrap();

    // evaluate algorithms
    alg_ode.evaluation().unwrap();
    alg_an.evaluation().unwrap();

    println!("ODE: \n\t-2ll: {:?}", alg_ode.n2ll());
    println!("Analytical: \n\t-2ll: {:?}", alg_an.n2ll());
    println!("=====================================");

    alg_ode.write_psi("examples/bimodal_ke/psi_ode.csv");
    alg_an.write_psi("examples/bimodal_ke/psi_an.csv");

    alg_ode.write_theta("examples/bimodal_ke/theta_ode.csv");
    alg_an.write_theta("examples/bimodal_ke/theta_an.csv");

    let psi_ode = alg_ode.psi().clone();
    let psi_an = alg_an.psi().clone();

    let ll_ode = burke(&psi_ode).unwrap();
    let ll_an = burke(&psi_an).unwrap();

    println!("ODE: \n\t-2ll: {:?}", -2. * ll_ode.1);
    println!("Analytical: \n\t-2ll: {:?}", -2. * ll_an.1);

    dbg!(&data.get_subjects().get(46));
}
