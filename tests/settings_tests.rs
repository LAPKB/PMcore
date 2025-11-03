use anyhow::Result;
use pmcore::prelude::*;

/// Test basic Settings builder construction
#[test]
fn test_settings_builder_basic() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0).add("v", 1.0, 20.0);

    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    // Test getters
    assert_eq!(settings.config().algorithm, Algorithm::NPAG);
    assert_eq!(settings.parameters().names().len(), 2);

    Ok(())
}

/// Test Settings serialization to JSON
#[test]
fn test_settings_serialization() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 15.0);

    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let settings = Settings::builder()
        .set_algorithm(Algorithm::NPOD)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    // Serialize to JSON
    let json = serde_json::to_string(&settings)?;

    // Should be valid JSON
    assert!(json.contains("\"algorithm\""));
    assert!(json.contains("\"parameters\""));

    // Deserialize back
    let deserialized: Settings = serde_json::from_str(&json)?;
    assert_eq!(deserialized.config().algorithm, Algorithm::NPOD);

    Ok(())
}

/// Test Settings with different algorithms
#[test]
fn test_settings_algorithms() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0);
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    // Test NPAG
    let settings_npag = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params.clone())
        .set_error_models(ems.clone())
        .build();
    assert_eq!(settings_npag.config().algorithm, Algorithm::NPAG);

    // Test NPOD
    let settings_npod = Settings::builder()
        .set_algorithm(Algorithm::NPOD)
        .set_parameters(params.clone())
        .set_error_models(ems.clone())
        .build();
    assert_eq!(settings_npod.config().algorithm, Algorithm::NPOD);

    Ok(())
}

/// Test Settings setters
#[test]
fn test_settings_setters() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0);
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    // Test set_cycles
    settings.set_cycles(50);
    assert_eq!(settings.config().cycles, 50);

    // Test set_algorithm
    settings.set_algorithm(Algorithm::NPOD);
    assert_eq!(settings.config().algorithm, Algorithm::NPOD);

    // Test set_cache
    settings.set_cache(false);
    assert_eq!(settings.config().cache, false);

    // Test set_idelta
    settings.set_idelta(0.5);
    assert_eq!(settings.predictions().idelta, 0.5);

    // Test set_tad
    settings.set_tad(24.0);
    assert_eq!(settings.predictions().tad, 24.0);

    Ok(())
}

/// Test Settings with prior
#[test]
fn test_settings_with_prior() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0);
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    // Set Sobol prior
    settings.set_prior(Prior::sobol(100, 42));

    // Verify prior was set using accessor methods
    assert_eq!(settings.prior().points(), Some(100));
    assert_eq!(settings.prior().seed(), Some(42));

    Ok(())
}

/// Test Settings with Latin Hypercube prior
#[test]
fn test_settings_latin_prior() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 15.0);
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    // Set Latin Hypercube prior
    settings.set_prior(Prior::Latin(50, 123));

    // Verify prior was set using accessor methods
    assert_eq!(settings.prior().points(), Some(50));
    assert_eq!(settings.prior().seed(), Some(123));

    Ok(())
}

/// Test Parameters functionality
#[test]
fn test_parameters() {
    let mut params = Parameters::new();

    // Add parameters
    params = params.add("ke", 0.1, 1.0);
    params = params.add("v", 5.0, 20.0);
    params = params.add("ka", 0.5, 2.0);

    // Check parameter count
    assert_eq!(params.names().len(), 3);

    // Check parameter names
    let names = params.names();
    assert!(names.contains(&"ke".to_string()));
    assert!(names.contains(&"v".to_string()));
    assert!(names.contains(&"ka".to_string()));
}

/// Test ErrorModels construction
#[test]
fn test_error_models() -> Result<()> {
    let em1 = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let em2 = ErrorModel::proportional(ErrorPoly::new(0.0, 0.0, 0.15, 0.0), 2.0);

    let mut ems = ErrorModels::new();
    ems = ems.add(0, em1)?;
    ems = ems.add(1, em2)?;

    // Should have 2 error models
    assert_eq!(ems.len(), 2);

    Ok(())
}

/// Test Config accessors
#[test]
fn test_config_accessors() -> Result<()> {
    let params = Parameters::new().add("ke", 0.1, 1.0);
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    let config = settings.config();

    // Test default values
    assert_eq!(config.algorithm, Algorithm::NPAG);
    assert!(config.cycles > 0);
    assert_eq!(config.cache, true);

    Ok(())
}
