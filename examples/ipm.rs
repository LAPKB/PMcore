use faer::mat;
use pmcore::{
    routines::evaluation::{ipm, ipm_old},
    structs::psi::Psi,
};
fn main() {
    let psi = ndarray::arr2(&[
        [0.1, 0.4, 0.4, 0.1],
        [0.5, 0.3, 0.1, 0.1],
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
    ]);
    eprintln!("Running old IPM");
    let a = ipm_old::burke(&psi).unwrap();
    dbg!(a);

    let psi = mat![
        [0.1, 0.4, 0.4, 0.1],
        [0.5, 0.3, 0.1, 0.1],
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25]
    ];
    let psi = Psi::from(psi.clone());
    eprintln!("Running new IPM");
    let a = ipm::burke(&psi).unwrap();
    dbg!(a);
}
