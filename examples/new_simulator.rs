use diffsol::vector::Vector;
use faer::{col, Col};
use pmcore::routines::data::{parse_pmetrics::read_pmetrics, DataTrait};
use pmcore::simulator::analytical::one_compartment_with_absorption;
use pmcore::{prelude::*, simulator::Equation};
use std::path::Path;
fn main() {
    let data = read_pmetrics(Path::new("examples/data/one_bimodal_ke.csv")).unwrap();
    let subjects = data.get_subjects();
    let first_subject = *subjects.first().unwrap();
    let spp = vec![3.1, 1.2, 70.0];

    let ode = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            //fetch_cov!(cov, t, creat);
            fetch_params!(p, ke, ka, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] + rateiv[0];
        },
        |_p, _t, _cov| col![0.0, 0.0],
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v);
            col![x[1] / v]
        },
    );

    let sim_ode = ode.simulate_subject(first_subject, &spp);

    let analytical = Equation::new_analytical(
        one_compartment_with_absorption,
        |_p, _t, _cov| col![0.0, 0.0],
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, _ka, v);
            col![x[1] / v]
        },
    );

    let sim_analytical = analytical.simulate_subject(first_subject, &spp);
    sim_ode.iter().zip(&sim_analytical).for_each(|(ode, anal)| {
        println!("ode : {}", ode);
        println!("anal: {}", anal);
        println!("=======================");
    })
}
