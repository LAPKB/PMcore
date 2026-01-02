#!/usr/bin/env Rscript
# SAEM Validation Reference Generator
# This script generates reference values from saemix for comparison with PMcore Rust implementation
#
# Usage: Rscript generate_reference.R
# Output: JSON files with reference values for each test case

library(saemix)
library(jsonlite)

# Get the script directory (works when run via Rscript)
get_script_dir <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- "--file="
    match <- grep(file_arg, args)
    if (length(match) > 0) {
        return(dirname(normalizePath(sub(file_arg, "", args[match]))))
    }
    return(getwd())
}

output_dir <- get_script_dir()
setwd(output_dir)

cat("=== SAEM Validation Reference Generator ===\n")
cat("Output directory:", getwd(), "\n\n")

# =============================================================================
# TEST CASE 1: One-Compartment IV Bolus (Simple)
# This matches the PMcore test_saem_convergence test
# =============================================================================

generate_one_compartment_iv_reference <- function() {
    cat("--- Test Case 1: One-Compartment IV Bolus ---\n")

    # True parameters (matching PMcore test)
    true_ke <- 0.4
    true_v <- 10.0
    dose <- 100.0

    # Generate synthetic data (20 subjects, same as PMcore)
    set.seed(42)
    n_subjects <- 20
    times <- c(0.5, 1.0, 2.0, 4.0, 8.0, 12.0)

    # Population variability (CV = 30%)
    omega_ke <- 0.09 # CV = sqrt(exp(0.09)-1) ≈ 30%
    omega_v <- 0.09

    # Residual error (additive, SD = 0.5)
    sigma <- 0.5

    # Create data frame
    data_rows <- list()
    for (id in 1:n_subjects) {
        # Random effects from normal distribution
        eta_ke <- rnorm(1, 0, sqrt(omega_ke))
        eta_v <- rnorm(1, 0, sqrt(omega_v))
        subj_ke <- true_ke * exp(eta_ke)
        subj_v <- true_v * exp(eta_v)

        for (t in times) {
            conc <- (dose / subj_v) * exp(-subj_ke * t)
            # Add residual error
            conc <- conc + rnorm(1, 0, sigma)
            # Ensure non-negative (censoring)
            conc <- max(conc, 0.01)
            data_rows[[length(data_rows) + 1]] <- data.frame(
                Id = id,
                Time = t,
                Dose = ifelse(t == times[1], dose, 0),
                Concentration = conc
            )
        }
    }

    sim_data <- do.call(rbind, data_rows)

    # Save data for Rust
    write.csv(sim_data, "onecomp_iv_data.csv", row.names = FALSE)
    cat("  Saved data to onecomp_iv_data.csv\n")

    # Create saemix data object
    saemix_data <- saemixData(
        name.data = sim_data,
        header = TRUE,
        name.group = c("Id"),
        name.predictors = c("Time"),
        name.response = c("Concentration"),
        name.X = "Time"
    )

    # One-compartment IV bolus model
    # C(t) = (Dose/V) * exp(-ke * t)
    model_1cpt_iv <- function(psi, id, xidep) {
        tim <- xidep[, 1]
        ke <- psi[id, 1]
        V <- psi[id, 2]
        # Dose is 100 for all subjects at t=0
        ypred <- (dose / V) * exp(-ke * tim)
        return(ypred)
    }

    # Model specification - log-normal for both parameters
    saemix_model <- saemixModel(
        model = model_1cpt_iv,
        description = "One-compartment IV bolus",
        psi0 = matrix(c(0.45, 10.0), # Initial values (geometric mean of bounds)
            ncol = 2, byrow = TRUE,
            dimnames = list(NULL, c("ke", "V"))
        ),
        transform.par = c(1, 1), # Both log-normal
        covariance.model = matrix(c(1, 0, 0, 1), ncol = 2, byrow = TRUE),
        omega.init = matrix(c(0.1, 0, 0, 0.1), ncol = 2, byrow = TRUE),
        error.model = "constant"
    )

    # SAEM options - use more iterations for better convergence
    saemix_options <- list(
        seed = 12345,
        nbiter.burn = 50, # More pure burn-in
        nbiter.saemix = c(200, 100), # 200 SA + 100 smoothing
        nb.chains = 3,
        nbiter.mcmc = c(3, 3, 3, 0), # More MCMC iterations per kernel
        proba.mcmc = 0.4,
        stepsize.rw = 0.4,
        alpha.sa = 0.97,
        rw.ini = 0.5,
        save = FALSE,
        save.graphs = FALSE,
        print = FALSE
    )

    # Run SAEM
    cat("  Running saemix...\n")
    saemix_fit <- saemix(saemix_model, saemix_data, saemix_options)
    saemix_fit <- map.saemix(saemix_fit)

    # Extract results
    results <- list(
        test_case = "one_compartment_iv",
        description = "One-compartment IV bolus with log-normal parameters",
        true_values = list(
            ke = true_ke,
            v = true_v
        ),

        # Population parameters in phi space (log-transformed)
        mu_phi = as.numeric(saemix_fit@results@fixed.effects),
        # Population parameters in psi space (original)
        mu_psi = as.numeric(exp(saemix_fit@results@fixed.effects)),

        # Covariance matrix
        omega = as.matrix(saemix_fit@results@omega),
        omega_diag = as.numeric(diag(saemix_fit@results@omega)),

        # Residual error
        sigma = as.numeric(saemix_fit@results@respar[1]),

        # Likelihood (linearization approximation)
        ll_lin = saemix_fit@results@ll.lin,
        objf = -2 * saemix_fit@results@ll.lin,

        # Individual estimates (MAP)
        map_psi = as.matrix(saemix_fit@results@map.psi),
        map_eta = as.matrix(saemix_fit@results@map.eta),
        cond_mean_phi = as.matrix(saemix_fit@results@cond.mean.phi),

        # Settings for reproducibility
        settings = list(
            seed = 12345,
            n_burn = 50,
            n_sa = 200,
            n_smooth = 100,
            n_chains = 3,
            transform_par = c(1, 1), # log-normal
            error_model = "constant",
            initial_psi = c(0.45, 10.0),
            initial_omega_diag = c(0.1, 0.1)
        ),

        # Data info
        n_subjects = n_subjects,
        n_observations = nrow(sim_data)
    )

    # Save results
    write_json(results, "onecomp_iv_reference.json", pretty = TRUE, auto_unbox = TRUE)
    cat("  Saved results to onecomp_iv_reference.json\n")

    # Print summary
    cat("\n  === Results Summary ===\n")
    cat("  True ke:", true_ke, "  Estimated:", results$mu_psi[1], "\n")
    cat("  True V:", true_v, "  Estimated:", results$mu_psi[2], "\n")
    cat("  Omega diagonal:", results$omega_diag, "\n")
    cat("  Sigma:", results$sigma, "\n")
    cat("  -2LL:", results$objf, "\n\n")

    return(results)
}

# =============================================================================
# TEST CASE 2: Theophylline (Standard Reference)
# This is the classic NLME example from saemix
# =============================================================================

generate_theophylline_reference <- function() {
    cat("--- Test Case 2: Theophylline (Standard) ---\n")

    # Load built-in theophylline data
    data(theo.saemix)

    # Save for Rust (need to restructure for PMcore format)
    # PMcore needs: ID, TIME, AMT/DOSE, DV, EVID, CMT
    theo_export <- data.frame(
        ID = theo.saemix$Id,
        TIME = theo.saemix$Time,
        DOSE = theo.saemix$Dose,
        DV = theo.saemix$Concentration,
        Weight = theo.saemix$Weight,
        Sex = theo.saemix$Sex
    )
    write.csv(theo_export, "theo_data.csv", row.names = FALSE)
    cat("  Saved data to theo_data.csv\n")

    # Create saemix data object
    theo_saemix_data <- saemixData(
        name.data = theo.saemix,
        header = TRUE,
        sep = " ",
        na = NA,
        name.group = c("Id"),
        name.predictors = c("Dose", "Time"),
        name.response = c("Concentration"),
        name.covariates = c("Weight", "Sex"),
        units = list(x = "hr", y = "mg/L", covariates = c("kg", "-")),
        name.X = "Time"
    )

    # One-compartment model with first-order absorption
    model_1cpt_oral <- function(psi, id, xidep) {
        dose <- xidep[, 1]
        tim <- xidep[, 2]
        ka <- psi[id, 1]
        V <- psi[id, 2]
        CL <- psi[id, 3]
        k <- CL / V
        ypred <- dose * ka / (V * (ka - k)) * (exp(-k * tim) - exp(-ka * tim))
        return(ypred)
    }

    # Model specification
    theo_model <- saemixModel(
        model = model_1cpt_oral,
        description = "One-compartment model with first-order absorption",
        psi0 = matrix(c(1.0, 20, 0.5),
            ncol = 3, byrow = TRUE,
            dimnames = list(NULL, c("ka", "V", "CL"))
        ),
        transform.par = c(1, 1, 1), # All log-normal
        covariance.model = matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), ncol = 3, byrow = TRUE),
        omega.init = matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), ncol = 3, byrow = TRUE),
        error.model = "constant"
    )

    # SAEM options
    theo_options <- list(
        seed = 12345,
        nbiter.burn = 5,
        nbiter.saemix = c(295, 100), # 300 burn-in/SA + 100 smoothing
        nb.chains = 3,
        nbiter.mcmc = c(2, 2, 2, 0),
        proba.mcmc = 0.4,
        stepsize.rw = 0.4,
        alpha.sa = 0.97,
        rw.ini = 0.5,
        save = FALSE,
        save.graphs = FALSE,
        print = FALSE
    )

    # Run SAEM
    cat("  Running saemix...\n")
    theo_fit <- saemix(theo_model, theo_saemix_data, theo_options)
    theo_fit <- map.saemix(theo_fit)

    # Extract results
    results <- list(
        test_case = "theophylline",
        description = "One-compartment oral absorption (ka, V, CL)",

        # Population parameters
        mu_phi = as.numeric(theo_fit@results@fixed.effects),
        mu_psi = as.numeric(exp(theo_fit@results@fixed.effects)),

        # Covariance
        omega = as.matrix(theo_fit@results@omega),
        omega_diag = as.numeric(diag(theo_fit@results@omega)),

        # Residual error
        sigma = as.numeric(theo_fit@results@respar[1]),

        # Likelihood
        ll_lin = theo_fit@results@ll.lin,
        objf = -2 * theo_fit@results@ll.lin,

        # Individual estimates
        map_psi = as.matrix(theo_fit@results@map.psi),
        map_eta = as.matrix(theo_fit@results@map.eta),
        cond_mean_phi = as.matrix(theo_fit@results@cond.mean.phi),

        # Settings
        settings = list(
            seed = 12345,
            n_burn = 5,
            n_sa = 295,
            n_smooth = 100,
            n_chains = 3,
            transform_par = c(1, 1, 1),
            error_model = "constant",
            initial_psi = c(1.0, 20.0, 0.5),
            initial_omega_diag = c(1.0, 1.0, 1.0)
        ),

        # Data info
        n_subjects = theo_saemix_data@N,
        n_observations = theo_saemix_data@ntot.obs
    )

    write_json(results, "theo_reference.json", pretty = TRUE, auto_unbox = TRUE)
    cat("  Saved results to theo_reference.json\n")

    cat("\n  === Results Summary ===\n")
    cat("  ka:", results$mu_psi[1], "  V:", results$mu_psi[2], "  CL:", results$mu_psi[3], "\n")
    cat("  Omega diagonal:", results$omega_diag, "\n")
    cat("  Sigma:", results$sigma, "\n")
    cat("  -2LL:", results$objf, "\n\n")

    return(results)
}

# =============================================================================
# TEST CASE 3: Bimodal Ke (PMcore Internal Dataset)
# Tests algorithm behavior with multimodal distributions
# =============================================================================

generate_bimodal_ke_reference <- function() {
    cat("--- Test Case 3: Bimodal Ke ---\n")

    # Check if data file exists
    data_file <- "../../../examples/bimodal_ke/bimodal_ke.csv"
    if (!file.exists(data_file)) {
        cat("  WARNING: bimodal_ke.csv not found, skipping this test case\n\n")
        return(NULL)
    }

    # Load PMcore bimodal_ke data
    bimodal_data <- read.csv(data_file)
    cat("  Loaded", nrow(bimodal_data), "rows from bimodal_ke.csv\n")

    # Restructure for saemix (needs specific column names)
    # Assume PMcore format has ID, TIME, DV, AMT, EVID, etc.
    # We need to adapt based on actual format

    cat("  Column names:", paste(names(bimodal_data), collapse = ", "), "\n")

    # This will depend on the actual format of bimodal_ke.csv
    # For now, skip if format is not compatible
    cat("  TODO: Implement bimodal_ke reference generation\n\n")
    return(NULL)
}

# =============================================================================
# Component-Level Tests
# These test individual components for exact matching
# =============================================================================

generate_component_tests <- function() {
    cat("--- Component-Level Reference Values ---\n")

    # Test 1: Parameter transformation verification
    cat("  1. Parameter transforms:\n")

    # Log-normal transform
    psi_values <- c(0.1, 0.5, 1.0, 2.0, 5.0)
    phi_values <- log(psi_values)
    cat(
        "     Log-normal: psi=", paste(psi_values, collapse = ","),
        " -> phi=", paste(round(phi_values, 6), collapse = ","), "\n"
    )

    # Test 2: Sufficient statistics computation
    cat("  2. Sufficient statistics:\n")

    # Simple 2-parameter example
    phi_samples <- matrix(c(
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    ), nrow = 3, byrow = TRUE)

    s1 <- colSums(phi_samples) # [9, 12]
    s2 <- t(phi_samples) %*% phi_samples # [[35, 44], [44, 56]]
    mu <- s1 / 3 # [3, 4]
    omega <- s2 / 3 - mu %*% t(mu) # Sample variance

    cat("     S1:", paste(s1, collapse = ","), "\n")
    cat("     S2 diag:", paste(diag(s2), collapse = ","), "\n")
    cat("     mu:", paste(mu, collapse = ","), "\n")
    cat("     omega diag:", paste(round(diag(omega), 6), collapse = ","), "\n")

    # Test 3: Step size schedule
    cat("  3. Step size schedule (n_burn=100, n_smooth=200):\n")

    n_burn <- 100
    n_smooth <- 200
    n_total <- n_burn + n_smooth

    # R saemix step sizes
    stepsize <- rep(1, n_total)
    stepsize[(n_burn + 1):n_total] <- 1 / (1:n_smooth)

    test_iters <- c(1, 50, 100, 101, 150, 200, 300)
    for (k in test_iters) {
        if (k <= n_total) {
            cat("     iter", k, ": gamma =", round(stepsize[k], 6), "\n")
        }
    }

    # Save component tests
    component_results <- list(
        transforms = list(
            log_normal = list(
                psi = psi_values,
                phi = phi_values
            )
        ),
        sufficient_stats = list(
            phi_samples = phi_samples,
            s1 = as.numeric(s1),
            s2 = as.matrix(s2),
            mu = as.numeric(mu),
            omega = as.matrix(omega)
        ),
        step_size = list(
            n_burn = n_burn,
            n_smooth = n_smooth,
            schedule = as.numeric(stepsize)
        )
    )

    write_json(component_results, "component_reference.json", pretty = TRUE, auto_unbox = TRUE)
    cat("  Saved to component_reference.json\n\n")

    return(component_results)
}

# =============================================================================
# Run all test case generators
# =============================================================================

main <- function() {
    results <- list()

    # Component tests (always run)
    results$components <- generate_component_tests()

    # Full algorithm tests
    results$onecomp_iv <- tryCatch(
        generate_one_compartment_iv_reference(),
        error = function(e) {
            cat("  ERROR:", e$message, "\n")
            return(NULL)
        }
    )

    results$theophylline <- tryCatch(
        generate_theophylline_reference(),
        error = function(e) {
            cat("  ERROR:", e$message, "\n")
            return(NULL)
        }
    )

    results$bimodal_ke <- tryCatch(
        generate_bimodal_ke_reference(),
        error = function(e) {
            cat("  ERROR:", e$message, "\n")
            return(NULL)
        }
    )

    cat("=== Reference Generation Complete ===\n")
    cat("Generated files:\n")
    cat("  - component_reference.json\n")
    if (!is.null(results$onecomp_iv)) cat("  - onecomp_iv_reference.json\n")
    if (!is.null(results$theophylline)) cat("  - theo_reference.json\n")

    return(results)
}

# Run if executed directly
if (!interactive()) {
    main()
}
