# FORTRAN NPAG Algorithm - Detailed Analysis

## File: NPAGFULLA.FOR

### Overview

This is the full NPAG implementation that performs iterative optimization to find the posterior distribution of pharmacokinetic parameters given patient data and a prior distribution.

---

## Main Algorithm Structure

### 1. INITIALIZATION (Lines ~165-240)

```fortran
VOLSPA calculation: Product of parameter ranges
NACTLAST = NACTVE (for error recovery)
ICYCLE = 0
IPRED = 11 + ICYCLE
JCOL = 0
NSTORE = 0
eps (resolution) = 0.2  (20% initially)
```

**Key Variables:**

- `NACTVE`: Number of active support points
- `CORDEN(K,J)`: K-th support point, J-th coordinate (J=1..NVAR), CORDEN(K,NVAR+1) = density
- `VOLSPA`: Volume of integration space
- `eps`: Grid resolution parameter (starts at 0.2, halved when objective improves slowly)

---

## 2. MAIN CYCLE LOOP (Label 10001, Lines ~279-1000)

Each cycle performs:

### 2.1 SUBJECT LOOP (Lines ~304-1000)

For each subject (though in NPAGFULL there's only 1 subject):

#### A. FILE PREPARATION (Lines ~304-329)

```fortran
REWIND(27)
CALL NEWWORK1  ! Converts steady-state dose indicators to 100 dose sets
                ! Reads file 27 → writes to file 37
```

#### B. READ SUBJECT DATA (Lines ~332-385)

```fortran
CALL FILREAD(...)
! Reads from file 37:
! - NOBSER: number of observation times
! - YO(I,J): observed values
! - C0, C1, C2, C3: assay noise coefficients
```

#### C. CALCULATE ASSAY STANDARD DEVIATIONS (Lines ~379-398)

```fortran
DO I=1,NOBSER
  DO J=1,NUMEQT
    Y = YO(I,J)
    IF (Y .NE. -99) THEN
      SIG(I,J) = C0(J) + C1(J)*Y + C2(J)*Y^2 + C3(J)*Y^3
      SIGFAC = SIGFAC * SIG(I,J)
    ENDIF
  ENDDO
ENDDO
OFAC = SIGFAC * (2.5066...^(NOBSER*NUMEQT - MISVAL))
```

**Note:** 2.5066... = SQRT(2\*PI)

#### D. LIKELIHOOD CALCULATION LOOP (Lines ~475-590, Label 800)

For each support point IG = NSTORE+1 to NACTVE:

```fortran
800 CONTINUE
  ! Get support point from CORDEN(IG, 1:NVAR)
  CALL MAKEVEC(...)  ! Combine random and fixed parameters

  CALL IDPC(...)  ! Calculate model predictions
  ! Returns W = sum of squared residuals:
  ! W = Σ[(YO(I,J) - H(I,J))/SIG(I,J)]^2

  ! Calculate P(YJ|X) for support point IG
  IF (W .GT. 22708) THEN
    PYJGX(IG) = 0
  ELSE
    PYJGX(IG) = DEXP(-0.5*W) / OFAC
  ENDIF
  WORKK = PYJGX(IG)

  ! Calculate P(X,YJ) = P(X) * P(YJ|X)
  WORK(IG) = CORDEN(IG,NVAR+1) * PYJGX(IG)
ENDDO

! Update NSTORE = NACTVE
```

#### E. INTEGRATE TO GET P(YJ) (Lines ~577-621)

```fortran
CALL NOTINT(VOLSPA, NGRID, NACTVE, WORK, MAXGRD, PYJ)
! Integrates P(X,YJ) over X-space to get marginal P(YJ)

IF (PYJ = 0) THEN
  ! All likelihoods too small → Error
  WRITE ERROR MESSAGE
  STOP
ENDIF
```

---

## 3. OPTIMIZATION PHASE (Lines ~627-767, "cgam5" section)

This section implements **gamma optimization** for error model parameters.

### 3.1 Base IPM Call (Lines ~631-640)

```fortran
IF (IERRMOD .EQ. 1) IGAMMA = 1

IF (MOD(IGAMMA, 3) .EQ. 1) THEN
  CALL emint(PSI, LDPSI, THETA, LDTHETA, NACTVE, NSUB, IJOB,
             X, DX, Y, DY, FOBJBASE, GAP, NVAR, KEEP, IHESS)

  IF (IHESS .EQ. -1) GO TO 900  ! Hessian error → stop

  NACTVE0 = NACTVE
  ! Save optimal solution to DENSTOR(1,4)
ENDIF
```

### 3.2 Gamma Plus (Lines ~687-704)

```fortran
IF (MOD(IGAMMA, 3) .EQ. 2) THEN
  ! Try gamma * (1 + gamdel)
  CALL emint(..., FOBJPLUS, ...)

  IF (IHESS .EQ. -1) GO TO 900

  IF (FOBJPLUS .GT. FOBJBASE) THEN
    ! Accept gamma_plus
    FOBJBASE = FOBJPLUS
    GAMDEL = GAMDEL * 4.0
  ENDIF
ENDIF
```

### 3.3 Gamma Minus (Lines ~725-747)

```fortran
IF (MOD(IGAMMA, 3) .EQ. 0) THEN
  ! Try gamma / (1 + gamdel)
  CALL emint(..., FOBJMINUS, ...)

  IF (IHESS .EQ. -1) GO TO 900

  IF (FOBJMINUS .GT. FOBJBASE) THEN
    ! Accept gamma_minus
    FOBJBASE = FOBJMINUS
    GAMDEL = GAMDEL * 4.0
  ENDIF
ENDIF
```

### 3.4 Update Gamma Delta (Lines ~752-767)

```fortran
GAMDEL = GAMDEL * 0.5
IF (GAMDEL .LT. 0.01) GAMDEL = 0.01

! Reset FORDEN(*,NVAR+1) to best of three solutions
! and normalize
```

---

## 4. STATISTICS & CONVERGENCE CHECK (Lines ~774-852)

### 4.1 Save State (Lines ~806-817)

```fortran
! Save CORDEN → CORDLAST
! Save NACTVE → NACTLAST
! (For recovery if next cycle has Hessian error)
```

### 4.2 Check Maximum Cycles (Lines ~835-852)

```fortran
IF (ICYCLE .GE. MAXCYC) THEN
  IMAXCYC = 1
  ! But continue to check convergence
ENDIF
```

---

## 5. CONTROL SECTION (Lines ~857-917)

### 5.1 Calculate Improvement (Lines ~857-860)

```fortran
XIMPROVE = |FOBJ - FLAST|
```

### 5.2 Refine Resolution (Lines ~862-866)

```fortran
IF (XIMPROVE .LE. THETA_G .AND. EPS .GT. THETA_E) THEN
  EPS = EPS / 2
ENDIF
```

**Constants:**

- `THETA_G = 1e-4`: Objective function convergence threshold
- `THETA_E = 1e-4`: Minimum resolution

### 5.3 Check Convergence (Lines ~875-917)

```fortran
IF (RESOLVE .LE. 0.0001) THEN
  ! Calculate checkbig (convergence metric)

  IF (DABS(CHECKBIG) .LE. 0.01) THEN
    ! CONVERGENCE ACHIEVED
    GO TO 900
  ELSE
    ! Start new major cycle
    RESOLVE = 0.2
  ENDIF
ENDIF

IF (IMAXCYC .EQ. 1) GO TO 900
```

---

## 6. EXPANSION PHASE (Lines ~920-967)

### 6.1 Adaptive Grid Expansion

```fortran
DO IPOINT = 1, NACTVEOLD
  PCUR = CORDEN(IPOINT, NVAR+1) / (2*NVAR + 1)

  ! Update original point
  CORDEN(IPOINT, NVAR+1) = PCUR

  ! For each dimension IVAR
  DO IVAR = 1, NVAR
    DEL = (AB(IVAR,2) - AB(IVAR,1)) * RESOLVE

    ! Try point at -eps
    IF (trial point valid) THEN
      CALL CHECKD(...)  ! Check minimum distance
      IF (ICLOSE .EQ. 0) THEN
        NACTVE = NACTVE + 1
        ! Add lower trial point
      ENDIF
    ENDIF

    ! Try point at +eps
    IF (trial point valid) THEN
      CALL CHECKD(...)
      IF (ICLOSE .EQ. 0) THEN
        NACTVE = NACTVE + 1
        ! Add upper trial point
      ENDIF
    ENDIF
  ENDDO
ENDDO

GO TO 10001  ! Start new cycle
```

---

## 7. ENDGAME (Lines ~975-1033, Label 900)

```fortran
900 CONTINUE
  ! Set NACTVE = NACTLAST (if Hessian error occurred)
  ! Set CORDEN = CORDLAST

  WRITE convergence status (ICONVERGE = 0, 1, 2, or 3)

  CLOSE(37)
  RETURN
```

---

## KEY SUBROUTINE: emint (Lines 5934-6510)

### Purpose

Interior Point Method to solve the EM optimization problem:

**Maximize:** `f(x) = Σ_i log(Σ_j ψ(i,j) * x(j))`

**Subject to:** `x(j) ≥ 0, Σ_j x(j) = 1`

### Key Algorithm Steps

#### 1. Initialization (Lines 5934-6170)

```fortran
SUBROUTINE EMINT(PSI, LDPSI, THETA, LDTHETA, NPOINT, NSUB, IJOB,
                 X, DX, Y, DY, FOBJ, GAP, NVAR, KEEP, IHESS)

! Check dimensions
IF (NSUB .GT. MAXSUBem) STOP
IF (NPOINT .GT. MAXACTem) STOP

! Initialize
DO J = 1, NSUB
  S = 0
  DO I = 1, NPOINT
    X(I) = 1.0
    S = S + PSI(J,I)
  ENDDO
  PSISUM(J) = S
  PTX(J) = S
  W(J) = 1.0 / S
ENDDO

! Calculate PTW = W' * PSI
SHRINK = 0
DO I = 1, NPOINT
  SUM = 0
  DO J = 1, NSUB
    SUM = SUM + PSI(J,I) * W(J)
  ENDDO
  Y(I) = SUM
  IF (SUM .GT. SHRINK) SHRINK = SUM
ENDDO
SHRINK = 2.0 * SHRINK

! Scale variables
DO I = 1, NPOINT
  X(I) = 1.0 * SHRINK
  Y(I) = Y(I) / SHRINK
  Y(I) = 1.0 - Y(I)
  MU = MU + X(I) * Y(I)
ENDDO
MU = MU / NPOINT

! Calculate R = EROW - W .* PTX
RMAX = -1e38
DO J = 1, NSUB
  W(J) = W(J) / SHRINK
  PTX(J) = PTX(J) * SHRINK
  RMAX = MAX(RMAX, ABS(1.0 - W(J) * PTX(J)))
ENDDO

GAP = 1.0
EPS = 1e-10
SIG = 0.0
```

#### 2. Main IPM Loop (Label 100, Lines 6184-6400)

```fortran
100 CONTINUE

  ! Check convergence
  CONVAL = MAX(MU, RMAX, GAP)
  IF (MU ≤ EPS .AND. RMAX ≤ EPS .AND. GAP ≤ EPS) GO TO 9000

  ITER = ITER + 1
  SMU = SIG * MU

  ! Build Hessian matrix
  DO J = 1, NSUB
    DO K = 1, NSUB
      HESS(J,K) = 0
    ENDDO
  ENDDO

  ! Outer product portion: HESS += (X/Y) * PSI' * PSI
  DO I = 1, NPOINT
    SCALE = X(I) / Y(I)
    DO J = 1, NSUB
      FACT = SCALE * PSI(J,I)
      DO K = J, NSUB
        HESS(K,J) = HESS(K,J) + FACT * PSI(K,I)
      ENDDO
    ENDDO
  ENDDO

  ! Make symmetric
  DO J = 1, NSUB-1
    DO K = J+1, NSUB
      HESS(J,K) = HESS(K,J)
    ENDDO
  ENDDO

  ! Diagonal portion: HESS(J,J) += PTX(J) / W(J)
  DO J = 1, NSUB
    HESS(J,J) = HESS(J,J) + PTX(J) / W(J)
  ENDDO

  ! Cholesky factorization
  CALL DPOTRF('L', NSUB, HESS, MAXSUBem, INFO)

  IF (INFO .NE. 0) THEN
    IHESS = -1
    WRITE ERROR MESSAGE
    RETURN
  ENDIF

  ! Construct RHS: DW(J) = 1/W(J) - Σ_i PSI(J,I) * SMU/Y(I)
  DO J = 1, NSUB
    SUM = 0
    DO I = 1, NPOINT
      SUM = SUM + PSI(J,I) * SMU / Y(I)
    ENDDO
    DW(J) = 1.0 / W(J) - SUM
  ENDDO

  ! Solve HESS * DW = RHS
  CALL DPOTRS('L', NSUB, 1, HESS, MAXSUBem, DW, NSUB, INFO)

  ! Compute DY and DX
  DO I = 1, NPOINT
    SUM = 0
    DO J = 1, NSUB
      SUM = SUM + PSI(J,I) * DW(J)
    ENDDO
    DY(I) = -SUM
    DX(I) = SMU/Y(I) - X(I) - DY(I) * X(I) / Y(I)
  ENDDO

  ! Calculate step lengths
  ALFPRI = -0.5
  DO I = 1, NPOINT
    IF (DX(I)/X(I) .LE. ALFPRI) ALFPRI = DX(I)/X(I)
  ENDDO
  ALFPRI = -1.0 / ALFPRI
  ALFPRI = MIN(1.0, 0.99995 * ALFPRI)

  ALFDUAL = -0.5
  DO I = 1, NPOINT
    IF (DY(I)/Y(I) .LE. ALFDUAL) ALFDUAL = DY(I)/Y(I)
  ENDDO
  ALFDUAL = -1.0 / ALFDUAL
  ALFDUAL = MIN(1.0, 0.99995 * ALFDUAL)

  ! Update variables
  MU = 0
  DO I = 1, NPOINT
    X(I) = X(I) + ALFPRI * DX(I)
    Y(I) = Y(I) + ALFDUAL * DY(I)
    MU = MU + X(I) * Y(I)
  ENDDO
  MU = MU / NPOINT

  ! Update PTX
  DO J = 1, NSUB
    SUM = 0
    DO I = 1, NPOINT
      SUM = SUM + PSI(J,I) * X(I)
    ENDDO
    PTX(J) = SUM
  ENDDO

  ! Update W
  DO J = 1, NSUB
    W(J) = W(J) + ALFDUAL * DW(J)
  ENDDO

  ! Compute RMAX
  RMAX = 0
  DO J = 1, NSUB
    RTEST = 1.0 - W(J) * PTX(J)
    IF (ABS(RTEST) .GT. RMAX) RMAX = ABS(RTEST)
  ENDDO

  ! Compute GAP
  SUMLOGW = 0
  SUMLGPTX = 0
  DO J = 1, NSUB
    SUMLOGW = SUMLOGW + LOG(W(J))
    SUMLGPTX = SUMLGPTX + LOG(PTX(J))
  ENDDO
  GAP = ABS(SUMLOGW + SUMLGPTX) / (1.0 + ABS(SUMLGPTX))

  ! Adjust SIG
  IF (MU .LT. EPS .AND. RMAX .GT. EPS) THEN
    SIG = 1.0
  ELSE
    C2 = 100.0
    TERM1 = (1.0 - ALFPRI)^2
    TERM2 = (1.0 - ALFDUAL)^2
    TERM3 = (RMAX - MU) / (RMAX + C2 * MU)
    TERM = MAX(TERM1, TERM2)
    TERM = MAX(TERM, TERM3)
    SIG = MIN(0.3, TERM)
  ENDIF

  ! Compute objective function
  SUMX = 0
  DO I = 1, NPOINT
    SUMX = SUMX + X(I)
  ENDDO
  FOBJ = 0
  DO J = 1, NSUB
    FOBJ = FOBJ + LOG(PTX(J) / SUMX)
  ENDDO

  GO TO 100
```

#### 3. Finalization (Label 9000, Lines 6400-6510)

```fortran
9000 CONTINUE

  ! Normalize X to sum to 1
  SUMX = 0
  DO I = 1, NPOINT
    SUMX = SUMX + X(I)
  ENDDO
  DO I = 1, NPOINT
    X(I) = X(I) / SUMX
  ENDDO

  IF (IJOB .EQ. 0) RETURN

  ! Condensation: Remove points with probability < max/1000
  XLIM = 0
  DO I = 1, NPOINT
    IF (X(I) .GT. XLIM) XLIM = X(I)
  ENDDO
  XLIM = XLIM * 1e-3

  ISUM = 0
  DO I = 1, NPOINT
    IF (X(I) .GT. XLIM) THEN
      ISUM = ISUM + 1
      LIST(ISUM) = I
      ! Move psi columns
      DO J = 1, NSUB
        PSI(J,ISUM) = PSI(J,I)
      ENDDO
      ! Move theta rows
      DO J = 1, NVAR
        THETA(ISUM,J) = THETA(I,J)
      ENDDO
      X(ISUM) = X(I)
    ENDIF
  ENDDO

  ! QR decomposition for further condensation
  JOB = 1
  ! Normalize PSI rows
  DO I = 1, ISUM
    DO J = 1, NSUB
      PSI(J,I) = PSI(J,I) / PSISUM(J)
    ENDDO
  ENDDO

  CALL DQRDC(PSI, LDPSI, NSUB, ISUM, Y, IPIVOT, DY, JOB)

  ! Count kept points based on QR
  KEEP = 0
  LIMLOOP = MIN(NSUB, ISUM)
  DO I = 1, LIMLOOP
    TEST = DNRM2(I, PSI(1,I), 1)
    IF (ABS(PSI(I,I) / TEST) .GE. 1e-8) KEEP = KEEP + 1
  ENDDO

  ! Sort IPIVOT
  IF (ISUM .GT. 1) THEN
    DO I = 1, KEEP-1
      DO J = I, KEEP
        IF (IPIVOT(I)*IPIVOT(J) .NE. 0 .AND.
            IPIVOT(I) .GT. IPIVOT(J)) THEN
          ITEMP = IPIVOT(I)
          IPIVOT(I) = IPIVOT(J)
          IPIVOT(J) = ITEMP
        ENDIF
      ENDDO
    ENDDO
  ENDIF

  ! Restore PSI and condense based on KEEP
  DO K = 1, KEEP
    J = IPIVOT(K)
    IF (J .NE. 0) THEN
      DO JJ = 1, NSUB
        PSI(JJ,K) = PSI(JJ,J)
      ENDDO
      DO JVAR = 1, NVAR
        THETA(K,JVAR) = THETA(J,JVAR)
      ENDDO
      W(K) = X(LIST(J))
    ENDIF
  ENDDO

  RETURN
END
```

---

## KEY HYPERPARAMETERS

| Parameter                 | Value         | Description                                     |
| ------------------------- | ------------- | ----------------------------------------------- |
| `THETA_E`                 | 1e-4          | Minimum grid resolution (convergence criterion) |
| `THETA_G`                 | 1e-4          | Objective function convergence threshold        |
| `THETA_F`                 | 1e-2          | Major cycle convergence criterion               |
| `THETA_D`                 | 1e-4          | Minimum distance between support points         |
| Initial `eps`             | 0.2           | Initial grid resolution (20% of range)          |
| `eps` threshold           | 1e-10         | IPM convergence tolerance                       |
| Lambda cutoff             | max/1000      | Points with λ < max(λ)/1000 are dropped         |
| QR threshold              | 1e-8          | Minimum R diagonal ratio to keep point          |
| `SHRINK`                  | 2 \* max(PTW) | Scaling factor for IPM initialization           |
| `ALFPRI`/`ALFDUAL` safety | 0.99995       | Step length safety factor                       |
| `SIG` max                 | 0.3           | Maximum centering parameter                     |
| Max W cutoff              | 22708         | Maximum exponent argument (-0.5\*W)             |

---

## CONVERGENCE CRITERIA

1. **IPM Convergence:**

   - `MU ≤ 1e-10` (duality gap)
   - `RMAX ≤ 1e-10` (primal feasibility)
   - `GAP ≤ 1e-10` (duality measure)

2. **NPAG Convergence:**

   - `|Δ FOBJ| ≤ 1e-4` AND
   - `eps ≤ 1e-4` AND
   - `|CHECKBIG| ≤ 0.01` (median parameter change)

3. **Stop Conditions:**
   - Convergence achieved
   - Maximum cycles reached
   - Hessian singularity
   - Stop file detected

---

## CONDENSATION STRATEGY

Performed inside `emint` when `IJOB ≠ 0`:

1. **Lambda filtering:** Remove points with `λ < max(λ)/1000`
2. **QR decomposition:** Normalize psi rows, perform QR with column pivoting
3. **Rank detection:** Keep points where `|R(i,i)| / ||R(:,i)|| ≥ 1e-8`
4. **Reorder:** Sort pivot indices to avoid collisions

---

## EXPANSION STRATEGY

For each existing point:

1. Divide its probability by `(2*NVAR + 1)`
2. For each dimension:
   - Try point at `+eps * range`
   - Try point at `-eps * range`
   - Only add if: within bounds, satisfies minimum distance criterion

This creates up to `2*NVAR` new points per existing point.
