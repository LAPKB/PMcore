library(saemix)
library(ggplot2)
library(gridExtra)
library(tidyverse)
library(here)
data(theo.saemix)

i_am("SAEMix_loop.r")


file.remove(here("SAEMix_output", "saemix_trace.csv"))

use_one <- TRUE

data <- theo.saemix

if (use_one == TRUE) {
  data <- dplyr::filter(data, Id < 3)
}

saemix.data<-saemixData(name.data=data,header=TRUE,sep=" ",na=NA,
  name.group=c("Id"),name.predictors=c("Dose","Time"),name.response=c("Concentration"),
  name.covariates=c("Weight","Sex"),units=list(x="hr",y="mg/L",covariates=c("kg","-")),
  name.X="Time")

model1cpt<-function(psi,id,xidep) {
  dose <- xidep[, 1]
  time <- xidep[, 2]
  ka <- psi[id, 1]
  V  <- psi[id, 2]
  ke <- psi[id, 3]
  dose * ka / (V * (ka - ke)) *
    (exp(-ke * time) - exp(-ka * time))
}


trial_loop <- function(ka, ke, v, trial_id) {
  cat("Starting trial number: ", as.character(trial_id), "\n")

  saemix.model <- saemixModel(
    model = model1cpt,
    psi0 = matrix(
      c(ka, v, ke),
      ncol = 3,
      byrow = TRUE,
      dimnames = list(NULL, c("ka", "V", "ke"))
    ),
    transform.par = c(1, 1, 1),
    covariance.model = diag(3),
    omega.init = diag(c(1, 1, 1)),
    error.model = "constant",
    verbose = FALSE
  )

  saemix.config = saemixControl(
    seed = 632545, 
    nb.chains = 25, 
    nbiter.mcmc = c(0, 2, 0, 0), 
    nbiter.burn = 100, 
    nbiter.saemix = c(300, 150),
    print = FALSE,
    save = FALSE,
    save.graphs = FALSE,
    directory = here("SAEMix_output")
  )

  saemix.fit<-saemix(saemix.model, saemix.data, saemix.config)

  saemix_trace <- as.data.frame(saemix.fit@results@allpar)[-1, c("ka", "V", "ke"), drop = FALSE]
  saemix_trace$cycle <- 1:nrow(saemix_trace)
  saemix_trace$trial_id <- trial_id

  col_names = FALSE
  if (trial_id == 0) {col_names = TRUE}

  write.table(saemix_trace, 
              file = here("SAEMix_output", "saemix_trace.csv"), 
              append = TRUE, 
              sep = ",", 
              col.names = col_names, 
              row.names = FALSE)
}

inits <- read.csv(here("random_init.csv"))

for (i in 1:nrow(inits)) {
  row <- inits[i, ]
  trial_loop(row$ka, row$ke, row$v, row$trial_id)
}
