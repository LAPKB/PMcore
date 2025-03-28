# Rust NPAG vs Fortran NPAGFULL: Comparative Analysis

## Architecture Overview

### Fortran NPAGFULL (Procedural)

- **Single monolithic subroutine** (~9000 lines)
- Sequential execution with GOTO statements
- Direct file I/O throughout
- Arrays passed via COMMON blocks and arguments

### Rust NPAG (Object-Oriented/Trait-Based)

- **Modular trait-based architecture**
- `NPAG` struct implementing `Algorithms` trait
- Structured error handling with `Result<T>`
- Immutable by default with explicit `&mut self`

---

## Core Differences

### 1. Algorithm Scope

| Aspect       | Fortran NPAGFULL         | Rust NPAG           |
| ------------ | ------------------------ | ------------------- |
| **Purpose**  | Single-subject posterior | Population analysis |
| **Subjects** | 1 (always)               | Multiple (1-N)      |
| **Cycles**   | Configurable (0-N)       | Always iterative    |
| **Output**   | Modified input arrays    | `NPResult` struct   |

**Key Insight**:

- Fortran NPAGFULL = **Subject-level** Bayesian calculator
- Rust NPAG = **Population-level** parameter estimator

### 2. Convergence Strategy

#### Fortran NPAGFULL (Adaptive Resolution)

```fortran
if(dabs(ximprove) .le. tol .and. resolve .gt. 0.0001) then
    resolve = resolve * 0.5
endif

if(resolve .le. 0.0001) then
    resolve = 0.2
    checkbig = fobj - prebig
    if(dabs(checkbig) .le. 0.01) then
        ! Converged
        go to 900
    endif
endif
```

**Strategy**: "Resolve refinement"

- Starts with `resolve = 0.2` (20% of parameter range)
- Halves when improvement small
- Resets to 0.2 when bottoms out at 0.0001
- Convergence when objective function change < 0.01

#### Rust NPAG (Epsilon-Based)

```rust
if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
    self.eps /= 2.;
    if self.eps <= THETA_E {
        let pyl = psi * w.weights();
        self.f1 = pyl.iter().map(|x| x.ln()).sum();
        if (self.f1 - self.f0).abs() <= THETA_F {
            // Converged
            self.converged = true;
        } else {
            self.f0 = self.f1;
            self.eps = 0.2;
        }
    }
}
```

**Strategy**: "Epsilon-based grid refinement"

- Starts with `eps = 0.2`
- Halves when `|Δobjf| ≤ THETA_G (1e-4)`
- When `eps ≤ THETA_E (1e-4)`, checks final convergence
- Convergence when `|f1 - f0| ≤ THETA_F (1e-2)`

**Similarities**: Both use adaptive parameter that gets halved and reset
**Differences**:

- Fortran: `resolve` controls grid expansion distance
- Rust: `eps` controls condensation tolerance

---

## 3. Grid Management

### Fortran NPAGFULL: Explicit Grid Operations

#### Expansion

```fortran
do ipoint = 1, nactveold
    pcur = corden(ipoint, nvar+1) / (2*nvar+1)
    corden(ipoint, nvar+1) = pcur

    do ivar = 1, nvar
        del = (ab(ivar,2) - ab(ivar,1)) * resolve

        ! Create trial point at -del
        corden(nactve+1, ivar) = corden(ipoint, ivar) - del
        call checkd(...)  ! Check minimum distance
        if(acceptable) nactve = nactve + 1

        ! Create trial point at +del
        corden(nactve+1, ivar) = corden(ipoint, ivar) + del
        call checkd(...)
        if(acceptable) nactve = nactve + 1
    enddo
enddo
```

**Creates**: `2 × nvar + 1` points per existing point (if distance checks pass)

#### Condensation

```fortran
call emint(...)  ! Interior point method
! emint sets denstor(i,1) = "keep" flags

! Later: filter based on keep flags
```

### Rust NPAG: Functional Grid Operations

#### Expansion

```rust
pub fn adaptative_grid(
    theta: &mut Theta,
    eps: f64,
    ranges: &[(f64, f64)],
    theta_d: f64
) {
    // Uses external adaptative_grid function
    // Adds points around promising regions
    // Controlled by eps parameter
}
```

#### Condensation

```rust
// Lambda-based filtering
let max_lambda = self.lambda.iter()
    .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

let mut keep = Vec::<usize>::new();
for (index, lam) in self.lambda.iter().enumerate() {
    if lam > max_lambda / 1000_f64 {
        keep.push(index);
    }
}
self.theta.filter_indices(keep.as_slice());

// QR decomposition-based filtering
let (r, perm) = qr::qrd(&self.psi)?;
let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());

for i in 0..keep_n {
    let ratio = r.get(i, i) / r.col(i).norm_l2();
    if ratio.abs() >= 1e-8 {
        keep.push(*perm.get(i).unwrap());
    }
}
self.theta.filter_indices(keep.as_slice());
```

**Two-stage condensation**:

1. **Lambda filtering**: Removes points with `λ < max(λ)/1000`
2. **QR decomposition**: Removes linearly dependent points

---

## 4. Optimization Method

### Fortran: Gamma Optimization (Error Model)

```fortran
igamma = igamma + 1

if(mod(igamma, 3) .eq. 1) then
    ! Try gamma_baseline
    call emint(...)
    fobjbase = fobj
endif

if(mod(igamma, 3) .eq. 2) then
    gamma = gammap  ! gamma * (1 + gamdel)
    call emint(...)
    fobjplus = fobj
endif

if(mod(igamma, 3) .eq. 0) then
    gamma = gammam  ! gamma / (1 + gamdel)
    call emint(...)
    fobjminu = fobj
endif

! Keep best gamma and adjust gamdel
```

**Approach**: Three-point search around current gamma

- Tests: baseline, +δ, -δ
- Adaptive step size (gamdel)
- Cycles through 3 evaluations

### Rust: Bracketing Optimization

```rust
let gamma_up = em.factor()? * (1.0 + self.gamma_delta[outeq]);
let gamma_down = em.factor()? / (1.0 + self.gamma_delta[outeq]);

let psi_up = calculate_psi(..., &error_model_up, ...)?;
let psi_down = calculate_psi(..., &error_model_down, ...)?;

let (lambda_up, objf_up) = burke(&psi_up)?;
let (lambda_down, objf_down) = burke(&psi_down)?;

if objf_up > self.objf {
    self.error_models.set_factor(outeq, gamma_up)?;
    self.objf = objf_up;
    self.gamma_delta[outeq] *= 4.;
}
if objf_down > self.objf {
    self.error_models.set_factor(outeq, gamma_down)?;
    self.objf = objf_down;
    self.gamma_delta[outeq] *= 4.;
}
self.gamma_delta[outeq] *= 0.5;
```

**Approach**: Two-point search with adaptive steps

- Tests: up, down (no baseline needed - already have current)
- Adaptive step size (gamma_delta)
- Multiplies by 4 on success, by 0.5 always
- Resets to 0.1 if < 0.01

---

## 5. Likelihood Calculation

### Fortran: Direct Loop Implementation

```fortran
do 800 jsub = 1, nsub
    ! Read subject data
    call filread(...)

    ! For each grid point
    do ig = 1, nactve
        ! Extract parameters
        ! Solve ODE
        call idpc(...)

        ! Calculate -0.5 * sum((obs - pred)²/σ²)
        work(ig) = p_yj_given_x
    enddo

    ! Integrate over parameter space
    call notint(work, pyj)

    ! Store pyjgx(jsub, :)
800 continue
```

### Rust: Functional/Parallel Implementation

```rust
self.psi = calculate_psi(
    &self.equation,
    &self.data,
    &self.theta,
    &self.error_models,
    show_progress,
    use_cache,
)?;

// In calculate_psi (external function):
subjects.par_iter()  // Parallel iteration
    .map(|subject| {
        theta.par_iter()  // Parallel over support points
            .map(|spp| {
                // Simulate
                let pred = equation.simulate_subject(subject, spp, cache)?;

                // Calculate likelihood
                observations.iter()
                    .map(|(obs, pred, errormodel)| {
                        errormodel.likelihood(obs, pred)
                    })
                    .product()  // Product of likelihoods
            })
            .collect()
    })
    .collect()
```

**Key Differences**:

- Fortran: Sequential, imperative loops
- Rust: Parallel (Rayon), functional chains
- Fortran: Accumulates in arrays
- Rust: Returns structured `Psi` object

---

## 6. Interior Point Method (IPM)

### Fortran: emint Subroutine

```fortran
subroutine emint(pyjgx, maxsub, corden, maxgrd, nactve, nsub,
                 iflag, den, keep, densold, densnew,
                 fobj, gap, nvar, keepout, ihess)
```

**Features**:

- **iflag = 1**: Initialize with IPM
- **iflag = 0**: Use previous solution as warm start
- Returns optimized densities in `den`
- Sets `keep` flags for condensation
- Detects singular Hessian (`ihess = -1`)

### Rust: burke Function

```rust
pub fn burke(psi: &Psi) -> Result<(Array1<f64>, f64)> {
    // Returns (lambda, objective_function)
    // Lambda = probability weights for support points
}
```

**Features**:

- Pure function (no side effects)
- Returns weights and objective
- Error handling via `Result`
- Does not modify input

**Major Difference**:

- Fortran `emint`: Stateful, modifies multiple arrays, flags for control flow
- Rust `burke`: Functional, returns values, errors via Result

---

## 7. Cycle Structure

### Fortran NPAGFULL Flow

```
[Read prior density]
    ↓
[If MAXCYC = 0] → [Calculate Bayesian posterior] → [Return]
    ↓
[Cycle 1..MAXCYC]
    ↓
[Evaluation] → [Gamma optimization] → [Condensation]
    ↓
[Check convergence]
    ↓
    If converged → [Return]
    If not → [Expansion] → [Next cycle]
```

### Rust NPAG Flow

```
[Initialize]
    ↓
[Loop: while !converged()]
    ↓
    [inc_cycle()]
    ↓
    [evaluation()] → Calculate psi
    ↓
    [condensation()] → Filter theta
    ↓
    [optimizations()] → Optimize error models
    ↓
    [convergence_evaluation()] → Check and update status
    ↓
    [logs()] → Write cycle info
    ↓
    [expansion()] → Add new support points
    ↓
[Return NPResult]
```

**Similarities**: Same logical flow
**Differences**:

- Fortran: GOTO-based, monolithic
- Rust: Method-based, modular, each step is separate function

---

## 8. Error Handling

### Fortran Strategy

```fortran
if(pyj .eq. 0.d0) then
    write(*, 26)
    open(47, file=errfil)
    write(47, 26)
    close(47)
    call pause
    stop
endif

if(ihess .eq. -1) go to 900  ! Jump to output creation
```

**Characteristics**:

- **Fatal errors**: STOP execution
- **Recoverable errors**: GOTO cleanup section
- **Error messages**: Written to screen and file
- **No propagation**: Each subroutine handles locally

### Rust Strategy

```rust
if let Err(err) = self.validate_psi() {
    bail!(err);
}

(self.lambda, _) = match burke(&self.psi) {
    Ok((lambda, objf)) => (lambda.into(), objf),
    Err(err) => {
        bail!("Error in IPM during evaluation: {:?}", err);
    }
};
```

**Characteristics**:

- **Result<T>**: All errors returned, not fatal
- **Error propagation**: `?` operator chains errors
- **Structured errors**: `anyhow::Error` with context
- **Graceful degradation**: Caller decides how to handle

---

## 9. Data Structures

### Fortran: Arrays and Common Blocks

```fortran
dimension corden(maxgrd, maxdim+1)
dimension pyjgx(maxsub, maxgrd)
dimension denstor(maxgrd, 4)
dimension work(maxgrd), workk(maxgrd)

common/cnst/ n, nd, ni, nup, nuic, np
common/cnst2/ npl, numeqt, ndrug, nadd
common/obser/ tim, sig, rs, yoo, bs
```

**Characteristics**:

- Fixed-size arrays
- Global state via COMMON
- No encapsulation
- Raw memory access

### Rust: Structured Types

```rust
pub struct NPAG<E: Equation> {
    equation: E,
    ranges: Vec<(f64, f64)>,
    psi: Psi,              // Matrix wrapper
    theta: Theta,          // Support points
    lambda: Weights,       // IPM output
    w: Weights,            // Final weights
    eps: f64,
    objf: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: ErrorModels,
    converged: bool,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    settings: Settings,
}
```

**Characteristics**:

- Encapsulated state
- Type-safe wrappers (`Psi`, `Theta`, `Weights`)
- Generic over equation type `<E: Equation>`
- Ownership and borrowing

---

## 10. Type Safety

### Fortran: Implicit Typing

```fortran
implicit real*8(a-h, o-z)  ! All variables a-h, o-z are double precision
dimension yo(594, maxnumeq)
```

**Issues**:

- Typos create new variables
- Array bounds not checked
- No compile-time guarantee of correctness

### Rust: Explicit Strong Typing

```rust
fn evaluation(&mut self) -> Result<()> {
    self.psi = calculate_psi(
        &self.equation,       // &E where E: Equation
        &self.data,           // &Data
        &self.theta,          // &Theta
        &self.error_models,   // &ErrorModels
        self.cycle == 1 && self.settings.config().progress,  // bool
        self.cycle != 1,      // bool
    )?;
    // ...
}
```

**Benefits**:

- Compile-time type checking
- No implicit conversions
- Generic constraints enforced
- Lifetime checking prevents dangling references

---

## 11. Memory Management

### Fortran

```fortran
! All arrays statically allocated or stack-based
! No dynamic allocation in NPAGFULL
! Memory freed at subroutine return
```

### Rust

```rust
// Stack for small fixed-size data
let eps: f64 = 0.2;

// Heap for dynamic/large data (automatic via Box, Vec)
let theta = Theta::new();  // Allocated on heap
// Automatically freed when theta goes out of scope (RAII)
```

**Key Difference**: Rust has automatic memory management without garbage collection

---

## 12. Parallelization

### Fortran NPAGFULL

- **Sequential execution**: Single-threaded
- **Parallelization**: External (compiler directives or OpenMP in other contexts)
- **In BestDose**: Multiple NPAGFULL calls could be parallelized externally

### Rust NPAG

```rust
subjects.par_iter()  // Rayon parallel iterator
    .map(|subject| {
        theta.par_iter()  // Nested parallelism
            .map(|spp| { /* ... */ })
```

**Built-in parallelism**:

- Rayon for data parallelism
- Thread-safe by design (ownership system)
- Nested parallelism: subjects × support points
- Work-stealing for load balancing

---

## 13. Testing and Validation

### Fortran

- **Testing**: Manual, example-based
- **Validation**: Compare output files
- **Debugging**: WRITE statements, debuggers

### Rust

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_condensation() {
        // Unit test for condensation
    }
}

// Integration tests in tests/ directory
// Benchmark tests with criterion
```

**Modern tooling**:

- Unit tests integrated
- Property-based testing (proptest)
- Automated CI/CD
- Benchmarking

---

## 14. Relationship Summary

### Fortran NPAGFULL ≈ Subset of Rust NPAG

**What Rust NPAG includes that NPAGFULL doesn't**:

- Multiple subject handling (population analysis)
- Parallel computation
- Structured error handling
- Modular architecture
- Type safety

**What NPAGFULL includes that Rust NPAG doesn't (yet)**:

- Zero-cycle Bayesian mode (NPAGFULL11 functionality)
- Three-point gamma search (Rust uses two-point)
- Some specific Fortran optimizations

### Could Rust NPAG Implement NPAGFULL11?

**Yes, easily**:

```rust
impl<E: Equation> NPAG<E> {
    fn bayesian_posterior(&mut self) -> Result<()> {
        // Just run evaluation once
        self.evaluation()?;

        // Filter to keep significant points
        let max_lambda = self.lambda.iter().max();
        let keep: Vec<_> = self.lambda.iter()
            .enumerate()
            .filter(|(_, &lam)| lam > max_lambda * 1e-100)
            .map(|(i, _)| i)
            .collect();

        self.theta.filter_indices(&keep);
        self.converged = true;
        Ok(())
    }
}
```

This would be the Rust equivalent of NPAGFULL11.

---

## Summary: Fortran vs Rust Design Philosophy

| Aspect          | Fortran (1960s-1990s)        | Rust (2010s-2020s)               |
| --------------- | ---------------------------- | -------------------------------- |
| **Paradigm**    | Procedural                   | Multi-paradigm (functional + OO) |
| **Safety**      | Programmer responsibility    | Compiler enforced                |
| **Errors**      | STOP or GOTO                 | Result<T> + ? operator           |
| **Concurrency** | External/manual              | Built-in + safe                  |
| **Memory**      | Manual/stack                 | Automatic (RAII)                 |
| **Testing**     | External scripts             | Integrated                       |
| **Modularity**  | Files + COMMON               | Modules + traits                 |
| **Performance** | Excellent (mature compilers) | Excellent (LLVM)                 |

Both implementations achieve the same mathematical goals but reflect very different programming eras and philosophies.
