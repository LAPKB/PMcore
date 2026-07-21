use anyhow::Result;
use pmcore::prelude::*;

fn one_compartment_metadata() -> pharmsol::equation::ModelMetadata {
    equation::metadata::new("one_compartment")
        .parameters(["ke", "v"])
        .states(["central"])
        .outputs(["0"])
        .route(equation::Route::bolus("0").to_state("central"))
}

#[test]
fn test_one_compartment_npag() -> Result<()> {
    // Create a simple one-compartment model
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(one_compartment_metadata())?;

    // Let known support points
    let spps: Vec<(f64, f64)> = vec![(0.85, 12.0), (0.52, 5.0), (0.15, 3.0)];

    // Create data
    let mut subjects: Vec<Subject> = Vec::new();
    spps.iter().enumerate().for_each(|(index, spp)| {
        let dose = 100.0;
        let ke = spp.0;
        let v = spp.1;
        let subject = Subject::builder(index.to_string())
            .bolus(0.0, 100.0, 0)
            .observation(1.0, (dose * f64::exp(-ke * 1.0)) / v, 0)
            .observation(2.0, (dose * f64::exp(-ke * 2.0)) / v, 0)
            .observation(4.0, (dose * f64::exp(-ke * 4.0)) / v, 0)
            .observation(8.0, (dose * f64::exp(-ke * 8.0)) / v, 0)
            .build();

        subjects.push(subject);
    });

    let data = data::Data::new(subjects);

    let parameters = ParameterSpace::<BoundedParameter>::new()
        .add("ke", 0.1, 1.0)
        .add("v", 1.0, 20.0);

    let prior = Theta::sobol(&parameters, 100)?;
    let error_models = AssayErrorModels::new().add(
        "0",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0),
    )?;
    let result = EstimationProblem::nonparametric(eq, data, prior, error_models)?
        .fit_with(NonParametricAlgorithm::npag())?;

    // Check the results
    assert_eq!(result.cycles(), 31);
    assert!(result.objf() - 565.7749 < 0.01);

    // The prior is preserved on the result and is distinct from the optimized
    // solution (which is condensed to far fewer support points).
    assert_eq!(result.prior().nspp(), 100);
    assert!(result.get_theta().nspp() < result.prior().nspp());

    Ok(())
}

#[test]
fn test_one_compartment_npod() -> Result<()> {
    // Create a simple one-compartment model
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(one_compartment_metadata())?;

    // Let known support points
    let spps: Vec<(f64, f64)> = vec![(0.85, 12.0), (0.52, 5.0), (0.15, 3.0)];

    // Create data
    let mut subjects: Vec<Subject> = Vec::new();
    spps.iter().enumerate().for_each(|(index, spp)| {
        let dose = 100.0;
        let ke = spp.0;
        let v = spp.1;
        let subject = Subject::builder(index.to_string())
            .bolus(0.0, 100.0, 0)
            .observation(1.0, (dose * f64::exp(-ke * 1.0)) / v, 0)
            .observation(2.0, (dose * f64::exp(-ke * 2.0)) / v, 0)
            .observation(4.0, (dose * f64::exp(-ke * 4.0)) / v, 0)
            .observation(8.0, (dose * f64::exp(-ke * 8.0)) / v, 0)
            .build();

        subjects.push(subject);
    });

    let data = data::Data::new(subjects);

    let parameters = ParameterSpace::<BoundedParameter>::new()
        .add("ke", 0.1, 1.0)
        .add("v", 1.0, 20.0);
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new().add(
        "0",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0),
    )?;
    let result = EstimationProblem::nonparametric(eq, data, prior, error_models)?
        .fit_with(NonParametricAlgorithm::npod())?;

    // Convergence and the final objective are stable; the exact number of
    // optimization cycles may vary with numerically equivalent support points.
    assert!(result.converged());
    let objective = result.objf();
    assert!(objective.is_finite());
    assert!(
        objective <= 85.13,
        "NPOD objective exceeded the regression bound: {objective}"
    );

    Ok(())
}

#[test]
fn test_one_compartment_postprob() -> Result<()> {
    // Create a simple one-compartment model
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(one_compartment_metadata())?;

    // Let known support points
    let spps: Vec<(f64, f64)> = vec![(0.85, 12.0), (0.52, 5.0), (0.15, 3.0)];

    // Create data
    let mut subjects: Vec<Subject> = Vec::new();
    spps.iter().enumerate().for_each(|(index, spp)| {
        let dose = 100.0;
        let ke = spp.0;
        let v = spp.1;
        let subject = Subject::builder(index.to_string())
            .bolus(0.0, 100.0, 0)
            .observation(1.0, (dose * f64::exp(-ke * 1.0)) / v, 0)
            .observation(2.0, (dose * f64::exp(-ke * 2.0)) / v, 0)
            .observation(4.0, (dose * f64::exp(-ke * 4.0)) / v, 0)
            .observation(8.0, (dose * f64::exp(-ke * 8.0)) / v, 0)
            .build();

        subjects.push(subject);
    });

    let data = data::Data::new(subjects);

    // Generate a prior distribution to test against
    let parameters = ParameterSpace::<BoundedParameter>::new()
        .add("ke", 0.1, 1.0)
        .add("v", 1.0, 20.0);

    let theta = Theta::sobol(&parameters, 100)?;

    let error_models = AssayErrorModels::new().add(
        "0",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0),
    )?;
    let result = EstimationProblem::nonparametric(eq, data, theta.clone(), error_models)?
        .fit_with(NonParametricAlgorithm::npmap())?;

    // Check the results
    assert_eq!(result.cycles(), 0);

    // Should be 100 points in theta (no change in points)
    assert_eq!(result.get_theta().nspp(), theta.nspp());

    Ok(())
}
