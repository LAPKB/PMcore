use anyhow::Result;
use pmcore::prelude::*;

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
    );

    let em = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, em).unwrap());

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

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

    let result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 100,
            prior: Some(Prior::sobol(64, 22)),
            ..RuntimeOptions::default()
        })
        .run()?;
    let result = result
        .as_nonparametric()
        .expect("NPAG should yield a nonparametric result");

    // Check the results
    assert_eq!(result.cycles(), 32);
    assert!(result.objf() - 565.7749 < 0.01);

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
    );

    let em = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, em).unwrap());

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

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

    let result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npod(
            NpodOptions::default(),
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 100,
            prior: Some(Prior::sobol(64, 22)),
            ..RuntimeOptions::default()
        })
        .run()?;
    let result = result
        .as_nonparametric()
        .expect("NPOD should yield a nonparametric result");

    // Check the results
    assert_eq!(result.cycles(), 11);
    assert!(result.objf() - 565.7749 < 0.01);

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
    );

    let em = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, em).unwrap());

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

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

    let result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(
            NonparametricMethod::Postprob(PostProbOptions::default()),
        ))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 100,
            prior: Some(Prior::sobol(64, 22)),
            ..RuntimeOptions::default()
        })
        .run()?;
    let result = result
        .as_nonparametric()
        .expect("POSTPROB should yield a nonparametric result");

    // Check the results
    assert_eq!(result.cycles(), 0);

    // Should be 64 points in theta (no change in points)
    assert_eq!(result.get_theta().nspp(), 64);

    Ok(())
}
