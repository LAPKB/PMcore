library(saemix)
library(ggplot2)
library(gridExtra)
library(tidyverse)
library(here)
data(theo.saemix)

i_am("SAEMix_loop.r")


file.remove(here("outputs", "SAEMix_output", "saemix_trace.csv"))

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
    c(1.0, 20, 0.1),
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
  directory = here("outputs", "SAEMix_output")
)


files <- list.files(path=here("test_data", "saemix_data"), pattern="*.csv", full.names=TRUE, recursive=FALSE)
trial_id <- 0

lapply(files, function(file) {
  saemix.data<-saemixData(name.data=file,header=TRUE,sep=",",na=NA,
    name.group=c("Id"),name.predictors=c("Dose","Time"),name.response=c("Concentration"),
    name.covariates=c("Weight","Sex"),units=list(x="hr",y="mg/L",covariates=c("kg","-")),
    name.X="Time")

  cat("Starting trial with file: ", file, "\n")

  saemix.fit<-saemix(saemix.model, saemix.data, saemix.config)

  saemix_trace <- as.data.frame(saemix.fit@results@allpar)[-1, c("ka", "V", "ke"), drop = FALSE]
  saemix_trace$cycle <- 1:nrow(saemix_trace)
  saemix_trace$data_path <- sub(paste0(".*", "examples/analytical_saem_test/test_data/saemix_data/") , "", file)

  col_names = FALSE
  if (trial_id == 0) {col_names = TRUE}

  write.table(saemix_trace, 
              file = here("outputs", "SAEMix_output", "saemix_trace.csv"), 
              append = TRUE, 
              sep = ",", 
              col.names = col_names, 
              row.names = FALSE)
    
  trial_id <- trial_id + 1
})
