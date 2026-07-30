library(saemix)
library(ggplot2)
library(gridExtra)
library(tidyverse)
data(theo.saemix)


use_one <- TRUE


data <- theo.saemix

if (use_one == TRUE) {
  data <- filter(data, Id < 3)
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

saemix.model <- saemixModel(
  model = model1cpt,
  psi0 = matrix(
    c(1, 20, 0.025),
    ncol = 3,
    byrow = TRUE,
    dimnames = list(NULL, c("ka", "V", "ke"))
  ),
  transform.par = c(1, 1, 1),
  covariance.model = diag(3),
  omega.init = diag(c(1, 1, 1)),
  error.model = "constant"
)

saemix.config = saemixControl(
  seed = 632545, 
  nb.chains = 25, 
  nbiter.mcmc = c(0, 2, 0, 0), 
  nbiter.burn = 100, 
  nbiter.saemix = c(300, 150),
  directory = "examples/analytical_saem_test/SAEMix_output"
)

saemix.fit<-saemix(saemix.model, saemix.data, saemix.config)
