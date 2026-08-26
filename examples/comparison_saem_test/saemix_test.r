library(saemix)
library(tidyverse)
library(here)

i_am("saemix_test.r")


data <- read.csv(here("data_saemix.csv"))
data <- cbind(data,tau=8)

saemix.data<-saemixData(
  name.data=data,header=TRUE,sep=" ",na=NA,
  name.group=c("Id"),name.predictors=c("Dose","Time","tau"),
  name.response=c("Concentration"),
  units=list(x="hr",y="mg/L"), name.X="Time")

model1cpt<-function(psi,id,xidep) {
  get_dose <- function(doses_applied, dose, time, tau, ka, ke, V) {
    if (max(unlist(doses_applied)) == 0) {
      return(doses_applied)
    }

    apply_new_dose <- doses_applied
    apply_new_dose[apply_new_dose > 0] <- 1
    doses_applied <- doses_applied - 1
    doses_applied[doses_applied < 0] <- 0

    (dose + (get_dose(doses_applied, dose, tau, tau, ka, ke, V) * apply_new_dose)) * 
      ka / (V * (ka - ke)) *
      (exp(-ke * time) - exp(-ka * time))
  }

  dose <- xidep[, 1]
  time <- xidep[, 2]
  tau  <- xidep[, 3]
  ka <- psi[id, 1]
  ke <- psi[id, 2]
  V  <- psi[id, 3]

  doses_applied <- time %/% tau
  time <- time - tau * doses_applied

  return(get_dose(doses_applied, tau, time, tau, ka, ke, V))
}

saemix.model <- saemixModel(
  model = model1cpt,
  psi0 = matrix(
    c(1.0, 0.1, 60.0),
    ncol = 3,
    byrow = TRUE,
    dimnames = list(NULL, c("ka", "ke", "V"))
  ),
  transform.par = c(1, 1, 1),
  omega.init = diag(c(1, 1, 1)),
  error.model = "proportional",
  error.init = c(0, 0.1),
  verbose = FALSE
)

saemix.config = saemixControl(
  seed = 632545, 
  nb.chains = 25, 
  # nbiter.mcmc = c(0, 2, 0, 0), 
  nbiter.burn = 5, 
  nbiter.saemix = c(300, 150),
  directory = here("outputs", "saemix_output"),
  save.graphs = FALSE,
  alpha.sa = 1.0,
)

saemix.fit<-saemix(saemix.model, saemix.data, saemix.config)
