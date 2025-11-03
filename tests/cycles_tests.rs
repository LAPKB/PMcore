use anyhow::Result;
use pmcore::algorithms::Status;
use pmcore::prelude::*;
use pmcore::routines::output::cycles::{CycleLog, NPCycle};
use pmcore::structs::theta::Theta;

/// Test NPCycle creation and accessors
#[test]
fn test_npcycle_creation() -> Result<()> {
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;

    let theta = Theta::new();

    let cycle = NPCycle::new(
        1,                 // cycle
        100.5,             // objf
        ems.clone(),       // error_models
        theta.clone(),     // theta
        10,                // nspp
        -5.2,              // delta_objf
        Status::Converged, // status
    );

    // Test accessors
    assert_eq!(cycle.cycle(), 1);
    assert_eq!(cycle.objf(), 100.5);
    assert_eq!(cycle.nspp(), 10);
    assert_eq!(cycle.delta_objf(), -5.2);

    Ok(())
}

/// Test NPCycle placeholder
#[test]
fn test_npcycle_placeholder() {
    let cycle = NPCycle::placeholder();

    // Placeholder should have default values
    assert_eq!(cycle.cycle(), 0);
    assert_eq!(cycle.objf(), 0.0);
    assert_eq!(cycle.nspp(), 0);
    assert_eq!(cycle.delta_objf(), 0.0);
}

/// Test CycleLog creation and operations
#[test]
fn test_cycle_log() -> Result<()> {
    let mut log = CycleLog::new();

    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;
    let theta = Theta::new();

    // Add a few cycles
    for i in 1..=5 {
        let cycle = NPCycle::new(
            i,
            100.0 - (i as f64) * 2.0,
            ems.clone(),
            theta.clone(),
            10 + i,
            -2.0,
            if i == 5 {
                Status::Converged
            } else {
                Status::InProgress
            },
        );
        log.push(cycle);
    }

    // Check that cycles were added
    assert_eq!(log.cycles().len(), 5);

    // Check individual cycles
    let cycles = log.cycles();
    assert_eq!(cycles[0].cycle(), 1);
    assert_eq!(cycles[4].cycle(), 5);

    Ok(())
}

/// Test CycleLog with different statuses
#[test]
fn test_cycle_log_statuses() -> Result<()> {
    let mut log = CycleLog::new();

    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let ems = ErrorModels::new().add(0, em)?;
    let theta = Theta::new();

    let statuses = vec![Status::Starting, Status::InProgress, Status::Converged];

    for (i, status) in statuses.iter().enumerate() {
        let cycle = NPCycle::new(
            i + 1,
            100.0,
            ems.clone(),
            theta.clone(),
            10,
            0.0,
            status.clone(),
        );
        log.push(cycle);
    }

    assert_eq!(log.cycles().len(), 3);

    Ok(())
}

/// Test Status enum display
#[test]
fn test_status_display() {
    let status_starting = Status::Starting;
    let status_progress = Status::InProgress;
    let status_converged = Status::Converged;
    let status_max = Status::MaxCycles;

    // These should be displayable
    let _ = format!("{:?}", status_starting);
    let _ = format!("{:?}", status_progress);
    let _ = format!("{:?}", status_converged);
    let _ = format!("{:?}", status_max);

    // Test Display trait
    assert!(format!("{}", status_converged).contains("Converged"));
}
