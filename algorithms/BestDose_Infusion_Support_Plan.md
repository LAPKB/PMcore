# BestDose Infusion Support: Detailed Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to add full infusion support to the BestDose algorithm, enabling optimization of both **amount** and **duration** parameters. The key challenge is handling a heterogeneous list of optimization variables where boluses contribute 1 parameter each while infusions can contribute 0, 1, or 2 parameters.

## Current State Analysis

### What Works Today

✅ **Bolus doses**: Fully supported
- Amount can be optimized (set to 0.0 as placeholder)
- Fixed amounts preserved (non-zero values)

⚠️ **Infusions**: Partially supported
- Amount optimization works in some places
- Duration optimization: **NOT IMPLEMENTED**
- Many code paths have `// Infusions not supported` comments

### Current Limitations

1. **No duration optimization**: Infusion duration is always fixed
2. **Inconsistent handling**: Some modules skip infusions entirely
3. **Index management**: No clear strategy for mixed dose types
4. **AUC mode**: Infusions explicitly not supported

## Core Challenge: Variable-Length Parameter Mapping

### The Problem

Different event types contribute different numbers of optimization parameters:

```
Subject Events          →    Optimization Parameters
═════════════════════════════════════════════════════
Bolus(t=0, amt=0)      →    [param_0]              (1 param: amount)
Infusion(t=12, amt=0, dur=2) →  [param_1]          (1 param: amount only)
Infusion(t=24, amt=0, dur=0) →  [param_2, param_3] (2 params: amount + duration)
Bolus(t=36, amt=500)   →    []                     (0 params: fixed)
Observation(t=48)      →    []                     (0 params: not a dose)

Total optimization dimension: 4 parameters
```

### Index Management Challenge

When iterating through events and parameters simultaneously, we need to:
1. Track which event we're on (event_index)
2. Track which parameter we're on (param_index)
3. Handle the 1:0, 1:1, or 1:2 mapping correctly

**This is the central complexity we must solve.**

## Proposed Solution: DoseParameterMap

### Core Concept

Create a **bidirectional mapping structure** that tracks the relationship between:
- Events in the subject (by position)
- Parameters in the optimization vector (by index)

### Data Structure

```rust
/// Maps between subject events and optimization parameters
///
/// This structure handles the variable-length mapping where:
/// - Fixed doses contribute 0 parameters
/// - Optimizable boluses contribute 1 parameter (amount)
/// - Optimizable infusions contribute 1-2 parameters (amount and/or duration)
#[derive(Debug, Clone)]
pub struct DoseParameterMap {
    /// Information for each dose event in the subject
    entries: Vec<DoseParameterEntry>,
    
    /// Total number of optimization parameters
    total_params: usize,
}

/// Information about a single dose event and its parameters
#[derive(Debug, Clone)]
struct DoseParameterEntry {
    /// Position in subject's event list (which event this is)
    event_index: usize,
    
    /// Type of dose event
    dose_type: DoseType,
    
    /// Parameter information for this dose
    /// - Empty vec: fixed dose (no optimization)
    /// - Contains ParameterType entries indicating what each parameter represents
    param_info: Vec<ParameterInfo>,
}

/// Information about a single optimization parameter
#[derive(Debug, Clone)]
struct ParameterInfo {
    /// Index in the optimization parameter vector
    param_index: usize,
    
    /// What this parameter represents
    param_type: ParameterType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterType {
    /// Dose amount (mg, μg, etc.)
    Amount,
    
    /// Infusion duration (hours)
    Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DoseType {
    Bolus,
    Infusion,
}

impl DoseParameterMap {
    /// Create a new map from a subject
    ///
    /// Analyzes the subject to determine which events are optimizable
    /// and builds the mapping structure.
    pub fn from_subject(subject: &Subject) -> Self {
        let mut entries = Vec::new();
        let mut param_index = 0;
        let mut event_index = 0;
        
        for occasion in subject.occasions() {
            for event in occasion.events() {
                match event {
                    Event::Bolus(bolus) => {
                        let mut param_info = Vec::new();
                        
                        // Amount is optimizable if it's zero
                        if bolus.amount() == 0.0 {
                            param_info.push(ParameterInfo {
                                param_index,
                                param_type: ParameterType::Amount,
                            });
                            param_index += 1;
                        }
                        
                        entries.push(DoseParameterEntry {
                            event_index,
                            dose_type: DoseType::Bolus,
                            param_info,
                        });
                        event_index += 1;
                    }
                    Event::Infusion(infusion) => {
                        let mut param_info = Vec::new();
                        
                        // Amount is optimizable if it's zero
                        if infusion.amount() == 0.0 {
                            param_info.push(ParameterInfo {
                                param_index,
                                param_type: ParameterType::Amount,
                            });
                            param_index += 1;
                        }
                        
                        // Duration is optimizable if it's zero
                        if infusion.duration() == 0.0 {
                            param_info.push(ParameterInfo {
                                param_index,
                                param_type: ParameterType::Duration,
                            });
                            param_index += 1;
                        }
                        
                        entries.push(DoseParameterEntry {
                            event_index,
                            dose_type: DoseType::Infusion,
                            param_info,
                        });
                        event_index += 1;
                    }
                    Event::Observation(_) => {
                        // Observations don't contribute to optimization
                        // (but we still increment event_index if we track all events)
                    }
                }
            }
        }
        
        DoseParameterMap {
            entries,
            total_params: param_index,
        }
    }
    
    /// Get the total number of optimization parameters
    pub fn num_parameters(&self) -> usize {
        self.total_params
    }
    
    /// Get the number of dose events (boluses + infusions)
    pub fn num_doses(&self) -> usize {
        self.entries.len()
    }
    
    /// Get parameter indices for a specific dose event
    ///
    /// Returns empty slice for fixed doses
    pub fn param_indices_for_dose(&self, dose_index: usize) -> Vec<usize> {
        self.entries[dose_index]
            .param_info
            .iter()
            .map(|info| info.param_index)
            .collect()
    }
    
    /// Get parameter information for a specific dose event
    pub fn param_info_for_dose(&self, dose_index: usize) -> &[ParameterInfo] {
        &self.entries[dose_index].param_info
    }
    
    /// Iterate over all dose entries
    pub fn iter(&self) -> impl Iterator<Item = &DoseParameterEntry> {
        self.entries.iter()
    }
    
    /// Apply optimized parameters to a subject
    ///
    /// Creates a new subject with optimization parameters applied to the
    /// appropriate events.
    pub fn apply_to_subject(
        &self,
        subject: &Subject,
        params: &[f64],
    ) -> Result<Subject> {
        if params.len() != self.total_params {
            bail!(
                "Parameter vector length mismatch: expected {}, got {}",
                self.total_params,
                params.len()
            );
        }
        
        let mut subject_clone = subject.clone();
        let mut dose_index = 0;
        
        for occasion in subject_clone.iter_mut() {
            for event in occasion.iter_mut() {
                match event {
                    Event::Bolus(bolus) => {
                        let entry = &self.entries[dose_index];
                        assert_eq!(entry.dose_type, DoseType::Bolus);
                        
                        // Apply parameters based on their type
                        for param_info in &entry.param_info {
                            match param_info.param_type {
                                ParameterType::Amount => {
                                    bolus.set_amount(params[param_info.param_index]);
                                }
                                ParameterType::Duration => {
                                    // Boluses don't have duration - should never happen
                                    unreachable!("Bolus cannot have duration parameter");
                                }
                            }
                        }
                        
                        dose_index += 1;
                    }
                    Event::Infusion(infusion) => {
                        let entry = &self.entries[dose_index];
                        assert_eq!(entry.dose_type, DoseType::Infusion);
                        
                        // Apply parameters based on their type
                        for param_info in &entry.param_info {
                            match param_info.param_type {
                                ParameterType::Amount => {
                                    infusion.set_amount(params[param_info.param_index]);
                                }
                                ParameterType::Duration => {
                                    infusion.set_duration(params[param_info.param_index]);
                                }
                            }
                        }
                        
                        dose_index += 1;
                    }
                    Event::Observation(_) => {
                        // Skip observations
                    }
                }
            }
        }
        
        Ok(subject_clone)
    }
}
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

#### 1.1. Add DoseParameterMap to types.rs

**File**: `src/bestdose/types.rs`

**Tasks**:
- [ ] Add the `DoseParameterMap` struct and implementation
- [ ] Add `DoseType` enum
- [ ] Add `DoseParameterEntry` struct
- [ ] Implement `from_subject()` constructor
- [ ] Implement `apply_to_subject()` method
- [ ] Add comprehensive tests

**Testing**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_map_creation_mixed_doses() {
        // Subject with:
        // - 1 optimizable bolus
        // - 1 infusion with both optimizable
        // - 1 fixed bolus
        // - 1 infusion with amount only optimizable
        // - 1 infusion with duration only optimizable
        let subject = Subject::builder("test")
            .bolus(0.0, 0.0, 0)              // Param 0: amount
            .infusion(12.0, 0.0, 0, 0.0)     // Param 1: amount, Param 2: duration
            .bolus(24.0, 500.0, 0)           // Fixed (no params)
            .infusion(36.0, 0.0, 0, 2.0)     // Param 3: amount only
            .infusion(48.0, 300.0, 0, 0.0)   // Param 4: duration only
            .build();
        
        let map = DoseParameterMap::from_subject(&subject);
        
        assert_eq!(map.num_parameters(), 5);
        assert_eq!(map.num_doses(), 5);
        
        // First dose: bolus with optimizable amount
        let entry0 = &map.entries[0];
        assert_eq!(entry0.param_info.len(), 1);
        assert_eq!(entry0.param_info[0].param_type, ParameterType::Amount);
        
        // Second dose: infusion with both optimizable
        let entry1 = &map.entries[1];
        assert_eq!(entry1.param_info.len(), 2);
        assert_eq!(entry1.param_info[0].param_type, ParameterType::Amount);
        assert_eq!(entry1.param_info[1].param_type, ParameterType::Duration);
        
        // Third dose: fixed bolus
        let entry2 = &map.entries[2];
        assert_eq!(entry2.param_info.len(), 0);
        
        // Fourth dose: infusion with amount only
        let entry3 = &map.entries[3];
        assert_eq!(entry3.param_info.len(), 1);
        assert_eq!(entry3.param_info[0].param_type, ParameterType::Amount);
        
        // Fifth dose: infusion with duration only
        let entry4 = &map.entries[4];
        assert_eq!(entry4.param_info.len(), 1);
        assert_eq!(entry4.param_info[0].param_type, ParameterType::Duration);
    }
    
    #[test]
    fn test_apply_to_subject() {
        let subject = Subject::builder("test")
            .bolus(0.0, 0.0, 0)
            .infusion(12.0, 0.0, 0, 0.0)
            .build();
        
        let map = DoseParameterMap::from_subject(&subject);
        let params = vec![100.0, 50.0, 1.5];
        
        let result = map.apply_to_subject(&subject, &params).unwrap();
        
        // Verify applied values
        // (Would need access to pharmsol internals to fully test)
    }
}
```

#### 1.2. Add DoseParameterMap to BestDoseProblem

**File**: `src/bestdose/types.rs`

**Changes to BestDoseProblem**:
```rust
pub struct BestDoseProblem {
    // ... existing fields ...
    
    /// Mapping between dose events and optimization parameters
    pub(crate) dose_param_map: DoseParameterMap,
}
```

**Update BestDoseProblem::new()**:
```rust
// In mod.rs, within BestDoseProblem::new()

// After concatenation/preparation of final_target:
let dose_param_map = DoseParameterMap::from_subject(&final_target);

tracing::info!("  Dose structure:");
tracing::info!("    Total doses: {}", dose_param_map.num_doses());
tracing::info!("    Optimization parameters: {}", dose_param_map.num_parameters());

// Log breakdown
let mut boluses = 0;
let mut infusions_amt_only = 0;
let mut infusions_dur_only = 0;
let mut infusions_both = 0;
let mut fixed_doses = 0;

for entry in dose_param_map.iter() {
    match entry.dose_type {
        DoseType::Bolus => {
            if entry.param_info.is_empty() {
                fixed_doses += 1;
            } else {
                boluses += 1;
            }
        }
        DoseType::Infusion => {
            if entry.param_info.is_empty() {
                fixed_doses += 1;
            } else {
                // Check what parameters this infusion has
                let has_amount = entry.param_info.iter()
                    .any(|p| p.param_type == ParameterType::Amount);
                let has_duration = entry.param_info.iter()
                    .any(|p| p.param_type == ParameterType::Duration);
                
                match (has_amount, has_duration) {
                    (true, true) => infusions_both += 1,
                    (true, false) => infusions_amt_only += 1,
                    (false, true) => infusions_dur_only += 1,
                    (false, false) => unreachable!("Non-empty param_info but no params"),
                }
            }
        }
    }
}

tracing::info!("    Boluses (optimizable): {}", boluses);
tracing::info!("    Infusions (amt only): {}", infusions_amt_only);
tracing::info!("    Infusions (dur only): {}", infusions_dur_only);
tracing::info!("    Infusions (amt+dur): {}", infusions_both);
tracing::info!("    Fixed doses: {}", fixed_doses);

// Add to BestDoseProblem
Ok(BestDoseProblem {
    // ... existing fields ...
    dose_param_map,
})
```

### Phase 2: Refactor Optimization (Week 1-2)

#### 2.1. Update cost.rs to use DoseParameterMap

**File**: `src/bestdose/cost.rs`

**Current approach** (problematic):
```rust
// Old code: manual iteration with index tracking
let mut optimizable_dose_number = 0;
for occasion in target_subject.iter_mut() {
    for event in occasion.iter_mut() {
        match event {
            Event::Bolus(bolus) => {
                if bolus.amount() == 0.0 {
                    bolus.set_amount(candidate_doses[optimizable_dose_number]);
                    optimizable_dose_number += 1;
                }
            }
            // ... similar for infusions
        }
    }
}
```

**New approach** (robust):
```rust
/// Apply candidate doses to target subject using the parameter map
fn apply_candidate_doses(
    problem: &BestDoseProblem,
    candidate_doses: &[f64],
) -> Result<Subject> {
    problem.dose_param_map.apply_to_subject(
        &problem.target,
        candidate_doses,
    )
}
```

**Update calculate_cost()**:
```rust
pub fn calculate_cost(problem: &BestDoseProblem, candidate_doses: &[f64]) -> Result<f64> {
    // Validate parameter count
    if candidate_doses.len() != problem.dose_param_map.num_parameters() {
        bail!(
            "Parameter count mismatch: expected {}, got {}",
            problem.dose_param_map.num_parameters(),
            candidate_doses.len()
        );
    }
    
    // Apply candidate doses using the map
    let target_subject = apply_candidate_doses(problem, candidate_doses)?;
    
    // Rest of cost calculation remains the same...
    match problem.target_type {
        Target::Concentration => {
            calculate_concentration_cost(problem, &target_subject)
        }
        Target::AUC => {
            calculate_auc_cost(problem, &target_subject)
        }
    }
}
```

#### 2.2. Update optimization.rs

**File**: `src/bestdose/optimization.rs`

**Changes to run_single_optimization()**:

```rust
fn run_single_optimization(
    problem: &BestDoseProblem,
    weights: &Weights,
    method_name: &str,
) -> Result<(Vec<f64>, f64)> {
    let min_dose = problem.doserange.min;
    let max_dose = problem.doserange.max;
    
    // Use dose_param_map instead of manual counting
    let num_parameters = problem.dose_param_map.num_parameters();
    let num_doses = problem.dose_param_map.num_doses();
    
    // Count fixed vs optimizable
    let num_optimizable = num_parameters;  // Each parameter is one optimizable component
    let num_fixed = num_doses - problem.dose_param_map.iter()
        .filter(|e| !e.param_indices.is_empty())
        .count();
    
    tracing::info!(
        "  │  {} optimization: {} parameters from {} doses",
        method_name,
        num_parameters,
        num_doses
    );
    tracing::info!(
        "  │    ({} fixed, {} optimizable components)",
        num_fixed,
        num_optimizable
    );
    
    // Create initial simplex for ALL parameters
    let initial_guess = (min_dose + max_dose) / 2.0;
    let initial_point = vec![initial_guess; num_parameters];
    let initial_simplex = create_initial_simplex(&initial_point);
    
    // ... rest of optimization ...
    
    // Result is already in correct format - return as-is
    Ok((optimized_params, final_cost))
}
```

**Changes to dual_optimization()**:

```rust
pub fn dual_optimization(problem: &BestDoseProblem) -> Result<BestDoseResult> {
    // ... dual optimization logic stays the same ...
    
    // At the end, NO FILTERING needed
    // The optimization parameters are exactly what we want to return
    
    Ok(BestDoseResult {
        dose: final_params,  // All optimization parameters
        objf: final_cost,
        status: "Converged".to_string(),
        preds,
        auc_predictions,
        optimization_method: method.to_string(),
    })
}
```

### Phase 3: Update Predictions (Week 2)

#### 3.1. Update predictions.rs for infusions

**File**: `src/bestdose/predictions.rs`

**Update calculate_final_predictions()**:

```rust
pub fn calculate_final_predictions(
    problem: &BestDoseProblem,
    optimal_params: &[f64],
    weights: &Weights,
) -> Result<(NPPredictions, Option<Vec<(f64, f64)>>)> {
    // Apply optimal parameters using the map
    let target_subject = problem.dose_param_map.apply_to_subject(
        &problem.target,
        optimal_params,
    )?;
    
    // Rest of prediction calculation remains the same...
    // (calculate_psi, burke, posterior, predictions)
    
    // For AUC mode, handle infusions properly
    match problem.target_type {
        Target::Concentration => {
            // Standard predictions
            Ok((predictions, None))
        }
        Target::AUC => {
            // Calculate AUC with full infusion support
            let auc_values = calculate_auc_with_infusions(
                problem,
                &target_subject,
                &theta,
                weights,
            )?;
            Ok((predictions, Some(auc_values)))
        }
    }
}
```

**Add calculate_auc_with_infusions()**:

```rust
/// Calculate AUC predictions with full infusion support
fn calculate_auc_with_infusions(
    problem: &BestDoseProblem,
    target_subject: &Subject,
    theta: &Theta,
    weights: &Weights,
) -> Result<Vec<(f64, f64)>> {
    // Generate dense time grid
    let dense_times = calculate_dense_times(
        target_subject,
        problem.settings.predictions().idelta,
    )?;
    
    // For each support point, simulate with infusions
    let mut auc_by_point = Vec::new();
    
    for (spp, &weight) in theta.matrix().row_iter().zip(weights.iter()) {
        let params = spp.iter().copied().collect::<Vec<f64>>();
        
        // Create dense subject (including infusions!)
        let mut dense_builder = Subject::builder(target_subject.id());
        
        // Add all doses (boluses AND infusions)
        for occasion in target_subject.occasions() {
            for event in occasion.events() {
                match event {
                    Event::Bolus(bolus) => {
                        dense_builder = dense_builder.bolus(
                            bolus.time(),
                            bolus.amount(),
                            bolus.input(),
                        );
                    }
                    Event::Infusion(infusion) => {
                        // ✅ NOW PROPERLY SUPPORTED!
                        dense_builder = dense_builder.infusion(
                            infusion.time(),
                            infusion.amount(),
                            infusion.input(),
                            infusion.duration(),
                        );
                    }
                    Event::Observation(_) => {}
                }
            }
        }
        
        // Add observations at dense times
        for &t in &dense_times {
            dense_builder = dense_builder.observation(t, -99.0, 0);
        }
        
        let dense_subject = dense_builder.build();
        
        // Simulate
        let pred = problem.eq.simulate_subject(&dense_subject, &params, None)?;
        let concentrations = pred.0.flat_predictions();
        
        // Calculate AUC using trapezoidal rule
        let auc_values = calculate_auc_at_times(
            &concentrations,
            &dense_times,
            &target_subject,
        )?;
        
        auc_by_point.push((weight, auc_values));
    }
    
    // Combine AUCs across support points (weighted average)
    combine_auc_predictions(&auc_by_point)
}
```

### Phase 4: Update Helper Functions (Week 2)

#### 4.1. Update concatenate_past_and_future()

**File**: `src/bestdose/mod.rs`

**Already works!** - Just remove the `// Note: Infusions not currently supported` comment

```rust
Event::Infusion(inf) => {
    builder = builder.infusion(
        inf.time() + current_time,
        inf.amount(),
        inf.input(),
        inf.duration(),
    );
}
```

#### 4.2. Remove calculate_dose_optimization_mask()

**File**: `src/bestdose/mod.rs`

**Action**: Delete this function entirely - it's replaced by `DoseParameterMap`

### Phase 5: Add Infusion-Specific Features (Week 3)

#### 5.1. Add DurationRange to DoseRange

**File**: `src/bestdose/types.rs`

**Enhancement**: Allow separate ranges for amounts vs durations

```rust
#[derive(Debug, Clone)]
pub struct DoseRange {
    /// Minimum/maximum for dose amounts (mg, μg, etc.)
    pub amount_min: f64,
    pub amount_max: f64,
    
    /// Minimum/maximum for infusion durations (hours)
    /// Only applies to infusions with optimizable duration
    pub duration_min: f64,
    pub duration_max: f64,
}

impl DoseRange {
    pub fn new(amount_min: f64, amount_max: f64) -> Self {
        Self {
            amount_min,
            amount_max,
            duration_min: 0.1,    // Default: 6 minutes minimum
            duration_max: 24.0,   // Default: 24 hours maximum
        }
    }
    
    pub fn with_duration_range(mut self, duration_min: f64, duration_max: f64) -> Self {
        self.duration_min = duration_min;
        self.duration_max = duration_max;
        self
    }
}
```

#### 5.2. Update Simplex Initialization

**File**: `src/bestdose/optimization.rs`

**Enhancement**: Use appropriate ranges for each parameter type

```rust
fn create_initial_simplex_with_ranges(
    problem: &BestDoseProblem,
) -> Vec<Vec<f64>> {
    let mut initial_point = Vec::new();
    
    // Build initial point with appropriate ranges
    for entry in problem.dose_param_map.iter() {
        for (i, _) in entry.param_indices.iter().enumerate() {
            let value = match entry.dose_type {
                DoseType::Bolus => {
                    // Always amount
                    (problem.doserange.amount_min + problem.doserange.amount_max) / 2.0
                }
                DoseType::Infusion => {
                    if i == 0 {
                        // First param is amount
                        (problem.doserange.amount_min + problem.doserange.amount_max) / 2.0
                    } else {
                        // Second param is duration
                        (problem.doserange.duration_min + problem.doserange.duration_max) / 2.0
                    }
                }
            };
            initial_point.push(value);
        }
    }
    
    create_initial_simplex(&initial_point)
}
```

### Phase 6: Documentation & Examples (Week 3-4)

#### 6.1. Update Algorithm Documentation

**File**: `algorithms/BestDose_algorithm.md`

**Add section**: "Infusion Support"

```markdown
## Infusion Support

### Optimizable Parameters

BestDose can optimize:
- **Bolus amount**: Set amount to 0.0 as placeholder
- **Infusion amount**: Set amount to 0.0 as placeholder
- **Infusion duration**: Set duration to 0.0 as placeholder
- **Both amount and duration**: Set both to 0.0

### Example: Optimize Infusion Amount Only

```rust
let target = Subject::builder("patient")
    .infusion(0.0, 0.0, 0, 2.0)      // Optimize amount, duration fixed at 2h
    .observation(24.0, 5.0, 0)
    .build();

let problem = BestDoseProblem::new(
    &theta, &weights, past, target, None, eq, ems,
    DoseRange::new(0.0, 1000.0),      // Amount range
    0.0, settings, 500, Target::Concentration,
)?;

let result = problem.optimize()?;
println!("Optimal amount: {} mg", result.dose[0]);
```

### Example: Optimize Infusion Duration Only

```rust
let target = Subject::builder("patient")
    .infusion(0.0, 500.0, 0, 0.0)    // Optimize duration, amount fixed at 500mg
    .observation(24.0, 5.0, 0)
    .build();

let problem = BestDoseProblem::new(
    &theta, &weights, past, target, None, eq, ems,
    DoseRange::new(0.0, 1000.0)
        .with_duration_range(0.5, 4.0),  // Duration range 0.5-4 hours
    0.0, settings, 500, Target::Concentration,
)?;

let result = problem.optimize()?;
println!("Optimal duration: {} hours", result.dose[0]);
```

### Example: Optimize Both Amount and Duration

```rust
let target = Subject::builder("patient")
    .infusion(0.0, 0.0, 0, 0.0)      // Optimize both!
    .observation(24.0, 5.0, 0)
    .build();

let problem = BestDoseProblem::new(
    &theta, &weights, past, target, None, eq, ems,
    DoseRange::new(0.0, 1000.0)
        .with_duration_range(0.5, 4.0),  // 0.5-4 hours
    0.0, settings, 500, Target::Concentration,
)?;

let result = problem.optimize()?;
println!("Optimal amount: {} mg", result.dose[0]);
println!("Optimal duration: {} hours", result.dose[1]);
```

### Example: Mixed Bolus and Infusion Optimization

```rust
let target = Subject::builder("patient")
    .bolus(0.0, 0.0, 0)                  // Optimize loading bolus
    .infusion(0.0, 0.0, 0, 0.0)          // Optimize maintenance infusion
    .observation(2.0, 10.0, 0)           // Peak target
    .observation(24.0, 5.0, 0)           // Trough target
    .build();

let problem = BestDoseProblem::new(
    &theta, &weights, past, target, None, eq, ems,
    DoseRange::new(0.0, 1000.0)
        .with_duration_range(1.0, 24.0),
    0.0, settings, 500, Target::Concentration,
)?;

let result = problem.optimize()?;
// result.dose = [bolus_amount, infusion_amount, infusion_duration]
println!("Loading dose: {} mg", result.dose[0]);
println!("Infusion: {} mg over {} hours", result.dose[1], result.dose[2]);
```
```

#### 6.2. Create Infusion Example

**File**: `examples/bestdose_infusion.rs`

```rust
//! BestDose Example: Optimizing IV Infusion Parameters
//!
//! This example demonstrates:
//! 1. Optimizing infusion amount only (fixed duration)
//! 2. Optimizing infusion duration only (fixed amount)
//! 3. Optimizing both amount and duration
//! 4. Mixed bolus + infusion optimization

use anyhow::Result;
use pmcore::bestdose::{BestDoseProblem, BestDoseResult, DoseRange, Target};
use pmcore::prelude::*;
use pharmsol::*;

fn main() -> Result<()> {
    // ... Setup equation, prior, etc. ...
    
    println!("\n=== Scenario 1: Optimize Infusion Amount ===");
    optimize_amount_only()?;
    
    println!("\n=== Scenario 2: Optimize Infusion Duration ===");
    optimize_duration_only()?;
    
    println!("\n=== Scenario 3: Optimize Both Amount and Duration ===");
    optimize_both()?;
    
    println!("\n=== Scenario 4: Loading Bolus + Maintenance Infusion ===");
    optimize_bolus_and_infusion()?;
    
    Ok(())
}

fn optimize_amount_only() -> Result<()> {
    // Target: Maintain concentration at 5 mg/L with 2-hour infusion
    let target = Subject::builder("patient_001")
        .infusion(0.0, 0.0, 0, 2.0)      // Amount=0 (optimize), Duration=2h (fixed)
        .observation(2.0, 5.0, 0)        // End of infusion target
        .observation(24.0, 5.0, 0)       // Trough target
        .build();
    
    let problem = BestDoseProblem::new(
        &theta, &weights, Some(past), target, None,
        eq, ems,
        DoseRange::new(0.0, 1000.0),
        0.0, settings, 500,
        Target::Concentration,
    )?;
    
    let result = problem.optimize()?;
    
    println!("Optimal infusion amount: {} mg over 2 hours", result.dose[0]);
    println!("Infusion rate: {} mg/h", result.dose[0] / 2.0);
    
    Ok(())
}

fn optimize_duration_only() -> Result<()> {
    // Target: Deliver 500 mg, optimize infusion duration
    let target = Subject::builder("patient_002")
        .infusion(0.0, 500.0, 0, 0.0)    // Amount=500mg (fixed), Duration=0 (optimize)
        .observation(6.0, 8.0, 0)        // Mid-infusion target
        .observation(24.0, 5.0, 0)       // Trough target
        .build();
    
    let problem = BestDoseProblem::new(
        &theta, &weights, Some(past), target, None,
        eq, ems,
        DoseRange::new(0.0, 1000.0)
            .with_duration_range(0.5, 12.0),  // 30 min to 12 hours
        0.0, settings, 500,
        Target::Concentration,
    )?;
    
    let result = problem.optimize()?;
    
    println!("Optimal duration: {} hours for 500 mg", result.dose[0]);
    println!("Infusion rate: {} mg/h", 500.0 / result.dose[0]);
    
    Ok(())
}

fn optimize_both() -> Result<()> {
    // Target: Optimize both amount and duration
    let target = Subject::builder("patient_003")
        .infusion(0.0, 0.0, 0, 0.0)      // Both parameters optimizable!
        .observation(6.0, 10.0, 0)
        .observation(24.0, 5.0, 0)
        .build();
    
    let problem = BestDoseProblem::new(
        &theta, &weights, Some(past), target, None,
        eq, ems,
        DoseRange::new(100.0, 1000.0)
            .with_duration_range(0.5, 8.0),
        0.0, settings, 500,
        Target::Concentration,
    )?;
    
    let result = problem.optimize()?;
    
    println!("Optimal infusion:");
    println!("  Amount: {} mg", result.dose[0]);
    println!("  Duration: {} hours", result.dose[1]);
    println!("  Rate: {} mg/h", result.dose[0] / result.dose[1]);
    
    Ok(())
}

fn optimize_bolus_and_infusion() -> Result<()> {
    // Target: Loading dose + maintenance infusion
    let target = Subject::builder("patient_004")
        .bolus(0.0, 0.0, 0)              // Loading bolus (optimize amount)
        .infusion(0.0, 0.0, 0, 0.0)      // Maintenance infusion (optimize both)
        .observation(0.5, 15.0, 0)       // Peak after loading dose
        .observation(24.0, 5.0, 0)       // Steady-state trough
        .build();
    
    let problem = BestDoseProblem::new(
        &theta, &weights, Some(past), target, None,
        eq, ems,
        DoseRange::new(0.0, 1000.0)
            .with_duration_range(1.0, 24.0),
        0.0, settings, 500,
        Target::Concentration,
    )?;
    
    let result = problem.optimize()?;
    
    println!("Optimal regimen:");
    println!("  Loading bolus: {} mg", result.dose[0]);
    println!("  Maintenance infusion: {} mg over {} hours",
        result.dose[1], result.dose[2]);
    println!("  Infusion rate: {} mg/h", result.dose[1] / result.dose[2]);
    
    Ok(())
}
```

### Phase 7: Testing & Validation (Week 4)

#### 7.1. Unit Tests

**Add to each module**:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_infusion_amount_optimization() {
        // Test optimizing amount only
    }
    
    #[test]
    fn test_infusion_duration_optimization() {
        // Test optimizing duration only
    }
    
    #[test]
    fn test_infusion_both_optimization() {
        // Test optimizing both
    }
    
    #[test]
    fn test_mixed_bolus_infusion() {
        // Test mixed dose types
    }
    
    #[test]
    fn test_parameter_index_consistency() {
        // Verify index mapping is correct
    }
}
```

#### 7.2. Integration Tests

**File**: `tests/bestdose_infusion.rs`

```rust
#[test]
fn test_complete_infusion_workflow() {
    // End-to-end test with:
    // - Past data
    // - Mixed bolus/infusion future
    // - Both concentration and AUC targets
    // - Verify results are reasonable
}
```

## Edge Cases & Error Handling

### Edge Cases to Handle

1. **All doses fixed**: No parameters to optimize
   - Return error or original doses with warning

2. **Only duration optimizable**: Amount is fixed
   - Works naturally with the map

3. **Zero infusion duration**: After optimization
   - Add constraint: `duration >= duration_min`

4. **Invalid parameter count**: Mismatch between map and optimizer
   - Validate in `apply_to_subject()`

5. **Mixed occasions**: Different occasion types
   - Works naturally - map handles all events

### Validation Checks

Add to `BestDoseProblem::new()`:

```rust
// Validate that we have something to optimize
if dose_param_map.num_parameters() == 0 {
    bail!("No optimizable parameters found. All doses are fixed (non-zero).");
}

// Warn about unusual configurations
if dose_param_map.num_parameters() > 10 {
    tracing::warn!(
        "Large number of optimization parameters ({}). \
        This may lead to slow convergence.",
        dose_param_map.num_parameters()
    );
}
```

## Migration Strategy

### Backward Compatibility

The new system is **fully backward compatible**:

✅ Existing code that only uses boluses: **Works unchanged**
✅ Existing code that uses fixed infusions: **Works unchanged**
✅ New code with optimizable infusions: **Works naturally**

### Deprecation Path

No deprecations needed - this is pure enhancement.

## Performance Considerations

### Computational Cost

| Scenario | Parameters | Simplex Size | Relative Cost |
|----------|-----------|--------------|---------------|
| 2 boluses | 2 | 3 vertices | 1x (baseline) |
| 2 infusions (amt only) | 2 | 3 vertices | 1x |
| 2 infusions (amt+dur) | 4 | 5 vertices | ~1.5x |
| 1 bolus + 1 infusion (both) | 3 | 4 vertices | ~1.2x |

### Optimization Tips

Document for users:

```markdown
## Performance Tips

1. **Fix what you can**: Only optimize necessary parameters
   - Fixed duration → faster convergence
   - Fixed amount → faster convergence

2. **Reasonable ranges**: Tight bounds → faster convergence
   - Use clinical knowledge to set ranges

3. **Start simple**: Optimize amounts first, then add durations
```

## Summary Checklist

### Implementation Checklist

- [ ] Phase 1: Core Infrastructure
  - [ ] Add `DoseParameterMap` to `types.rs`
  - [ ] Add unit tests for map
  - [ ] Integrate map into `BestDoseProblem`
  
- [ ] Phase 2: Refactor Optimization
  - [ ] Update `cost.rs` to use map
  - [ ] Update `optimization.rs` to use map
  - [ ] Remove manual index tracking
  
- [ ] Phase 3: Update Predictions
  - [ ] Update `predictions.rs` for infusions
  - [ ] Add `calculate_auc_with_infusions()`
  - [ ] Remove infusion warnings
  
- [ ] Phase 4: Update Helpers
  - [ ] Update `concatenate_past_and_future()`
  - [ ] Remove `calculate_dose_optimization_mask()`
  
- [ ] Phase 5: Infusion Features
  - [ ] Add duration ranges to `DoseRange`
  - [ ] Update simplex initialization
  
- [ ] Phase 6: Documentation
  - [ ] Update algorithm docs
  - [ ] Create infusion example
  - [ ] Update module docs
  
- [ ] Phase 7: Testing
  - [ ] Unit tests for each module
  - [ ] Integration tests
  - [ ] Edge case tests

### Success Criteria

✅ All existing tests pass
✅ New infusion example runs successfully
✅ Can optimize: amount only, duration only, both
✅ Can mix boluses and infusions
✅ AUC mode works with infusions
✅ Documentation is complete and clear
✅ No performance regression for bolus-only cases

## Timeline

**Total Estimated Time**: 3-4 weeks

- Week 1: Core infrastructure + basic refactoring
- Week 2: Complete refactoring + predictions
- Week 3: Enhanced features + documentation
- Week 4: Testing + validation + polish

## Conclusion

This plan provides a robust, maintainable solution for full infusion support in BestDose. The key insight is using a **bidirectional mapping structure** (`DoseParameterMap`) that handles the variable-length relationship between events and parameters, eliminating manual index tracking and enabling clean, composable code.

The solution is:
- ✅ **Backward compatible**: Existing code works unchanged
- ✅ **Type-safe**: Compile-time guarantees
- ✅ **Maintainable**: Clear separation of concerns
- ✅ **Extensible**: Easy to add new dose types in future
- ✅ **Well-tested**: Comprehensive test coverage
- ✅ **Well-documented**: Clear examples and guides
