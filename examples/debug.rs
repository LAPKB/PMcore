use pmcore::prelude::*;

fn main() -> Result<()> {
    let ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
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
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 2.0, None);
    let ems = ErrorModels::new().add(0, em).unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.initialize_logs()?;
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;

    let mut alg_ode = dispatch_algorithm(settings.clone(), ode, data.clone())?;
    let mut alg_an = dispatch_algorithm(settings.clone(), an, data.clone())?;

    // Initialize and run algorithms
    alg_ode.initialize()?;
    alg_an.initialize()?;

    // evaluate algorithms
    alg_ode.evaluation()?;
    alg_an.evaluation()?;

    println!("ODE: \n\t-2ll: {:?}", alg_ode.n2ll());
    println!("Analytical: \n\t-2ll: {:?}", alg_an.n2ll());
    println!("=====================================");

    alg_ode.psi().write("examples/bimodal_ke/psi_ode_full.csv");

    alg_ode.psi().write("examples/bimodal_ke/psi_ode.csv");
    alg_an.psi().write("examples/bimodal_ke/psi_an.csv");

    alg_ode.psi().write("examples/bimodal_ke/theta_ode.csv");
    alg_an.psi().write("examples/bimodal_ke/theta_an.csv");

    dbg!(&data.subjects().get(46));

    Ok(())
}
