# Run bimodal_ke dataset with R saemix
# This script tests the R SAEM implementation on the bimodal_ke dataset
# to compare with the Rust implementation

library(saemix)

# Read and prepare the data
raw_data <- read.csv("../bimodal_ke/bimodal_ke.csv")

# Filter observation records (EVID == 0) and get necessary columns
obs_data <- raw_data[raw_data$EVID == 0, ]

# saemix needs: ID, TIME, dose, observation
# Get doses for each subject
dose_data <- raw_data[raw_data$EVID == 1, c("ID", "DOSE")]
names(dose_data) <- c("ID", "DOSE")

# Merge dose with observations
saemix_data <- merge(obs_data[, c("ID", "TIME", "OUT")], dose_data, by = "ID")
saemix_data <- saemix_data[order(saemix_data$ID, saemix_data$TIME), ]
names(saemix_data) <- c("id", "time", "conc", "dose")

# Ensure numeric types
saemix_data$id <- as.integer(saemix_data$id)
saemix_data$time <- as.numeric(saemix_data$time)
saemix_data$conc <- as.numeric(saemix_data$conc)
saemix_data$dose <- as.numeric(saemix_data$dose)

cat("Data summary:\n")
cat("Number of subjects:", length(unique(saemix_data$id)), "\n")
cat("Total observations:", nrow(saemix_data), "\n")
cat("\nData types:\n")
print(sapply(saemix_data, class))
cat("\nFirst 20 rows:\n")
print(head(saemix_data, 20))

# Create saemix data object
saemix.data <- saemixData(
    name.data = saemix_data,
    header = TRUE,
    name.group = c("id"),
    name.predictors = c("dose", "time"),
    name.response = c("conc")
)

# Define the one-compartment IV bolus model
# For IV bolus: C = (Dose/V) * exp(-ke * t)
# Note: This dataset uses a 0.5h infusion, but we'll approximate as bolus
# The model: C = (Dose/V) * exp(-ke * t)
one_cpt_model <- function(psi, id, xidep) {
    dose <- xidep[, 1]
    time <- xidep[, 2]
    ke <- psi[id, 1]
    V <- psi[id, 2]

    # One compartment with IV bolus
    ypred <- (dose / V) * exp(-ke * time)
    return(ypred)
}

# Create saemix model
# psi0: initial estimates [ke, V]
# transform.par: 1 = log transform (lognormal distribution)
# NPAG found: ke mean=0.191, v mean=107
saemix.model <- saemixModel(
    model = one_cpt_model,
    modeltype = "structural",
    description = "One-compartment IV bolus model",
    psi0 = matrix(c(0.2, 110),
        ncol = 2, byrow = TRUE,
        dimnames = list(NULL, c("ke", "V"))
    ),
    transform.par = c(1, 1), # 1 = log-transform (lognormal)
    covariance.model = matrix(c(1, 0, 0, 1), ncol = 2, byrow = TRUE),
    omega.init = matrix(c(0.5, 0, 0, 0.5), ncol = 2, byrow = TRUE),
    error.model = "proportional" # gamma * ypred
)

# Run SAEM
cat("\n\nRunning SAEM algorithm...\n\n")
saemix.fit <- saemix(
    saemix.model,
    saemix.data,
    list(
        seed = 12345,
        nbiter.saemix = c(300, 100), # K1 burn-in, K2 estimation
        nb.chains = 3,
        print = TRUE,
        save = FALSE,
        save.graphs = FALSE
    )
)

# Print results
cat("\n\n========== SAEM Results ==========\n")
print(saemix.fit)

# Get population parameters
cat("\n--- Population Parameters ---\n")
cat("Fixed effects (mu):\n")
print(saemix.fit@results@fixed.effects)
cat("\nOmega (variance of random effects):\n")
print(saemix.fit@results@omega)
cat("\nResidual error:\n")
print(saemix.fit@results@respar)

# Get individual parameters
cat("\n--- Individual Parameters (first 10 subjects) ---\n")
indiv_params <- psi(saemix.fit)
print(head(indiv_params, 10))

# Summary statistics of individual ke
cat("\n--- Summary of individual ke estimates ---\n")
print(summary(indiv_params$ke))

# Check for bimodality
cat("\n--- Distribution of ke ---\n")
cat("Mean:", mean(indiv_params$ke), "\n")
cat("Median:", median(indiv_params$ke), "\n")
cat("SD:", sd(indiv_params$ke), "\n")
cat("Min:", min(indiv_params$ke), "\n")
cat("Max:", max(indiv_params$ke), "\n")

# Objective function
cat("\n--- Objective Function ---\n")
cat("-2LL:", -2 * saemix.fit@results@ll.is, "\n")
