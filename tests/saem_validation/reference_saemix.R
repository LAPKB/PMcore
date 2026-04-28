# Reference SAEM implementation using saemix R package
# This script runs the theophylline example and saves results for comparison

library(saemix)

# Load theophylline data
data(theo.saemix)

# Create saemix data object
saemix.data <- saemixData(
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
model1cpt <- function(psi, id, xidep) {
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
# Initial values: ka=1, V=20, CL=0.5
# All parameters log-transformed (transform.par=c(1,1,1))
# Diagonal covariance matrix for random effects
saemix.model <- saemixModel(
    model = model1cpt,
    description = "One-compartment model with first-order absorption",
    psi0 = matrix(c(1., 20, 0.5),
        ncol = 3, byrow = TRUE,
        dimnames = list(NULL, c("ka", "V", "CL"))
    ),
    transform.par = c(1, 1, 1), # Log transform all parameters
    covariance.model = matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), ncol = 3, byrow = TRUE),
    omega.init = matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), ncol = 3, byrow = TRUE),
    error.model = "constant"
)

# SAEM options
# algorithm: c(nburning, nexploration, nsmoothing)
# Using f-SAEM style with burn-in then SA
saemix.options <- list(
    algorithm = c(1, 1, 1), # Run SAEM
    seed = 12345,
    nbiter.saemix = c(300, 100), # 300 burn-in, 100 SA iterations
    nb.chains = 3,
    save = FALSE,
    save.graphs = FALSE,
    print = FALSE
)

# Run SAEM
cat("Running saemix...\n")
saemix.fit <- saemix(saemix.model, saemix.data, saemix.options)

# Extract results
cat("\n=== SAEMIX Results ===\n")
cat("\nPopulation parameters (fixed effects on log scale):\n")
print(saemix.fit@results@fixed.effects)

cat("\nPopulation parameters (original scale):\n")
psi_pop <- exp(saemix.fit@results@fixed.effects)
names(psi_pop) <- c("ka", "V", "CL")
print(psi_pop)

cat("\nRandom effect variances (omega^2):\n")
omega <- saemix.fit@results@omega
print(diag(omega))

cat("\nResidual error (sigma):\n")
print(saemix.fit@results@respar)

cat("\nObjective function (-2LL):\n")
print(saemix.fit@results@ll.lin * -2)

cat("\nIndividual parameters (first 5 subjects):\n")
# Get MAP estimates
saemix.fit <- map.saemix(saemix.fit)
head(saemix.fit@results@map.psi, 5)

# Save results for comparison
results <- list(
    mu_log = as.numeric(saemix.fit@results@fixed.effects),
    mu = as.numeric(exp(saemix.fit@results@fixed.effects)),
    omega_diag = as.numeric(diag(saemix.fit@results@omega)),
    sigma = as.numeric(saemix.fit@results@respar[1]),
    objf = as.numeric(saemix.fit@results@ll.lin * -2)
)

# Save to JSON for Rust test to read
library(jsonlite)
write_json(results, "saemix_results.json", pretty = TRUE, auto_unbox = TRUE)
cat("\nResults saved to saemix_results.json\n")

# Also save the data in a format the Rust test can read
theo_data <- theo.saemix[, c("Id", "Dose", "Time", "Concentration")]
write.csv(theo_data, "theo_data.csv", row.names = FALSE)
cat("Data saved to theo_data.csv\n")
