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
  dose<-xidep[,1]
  tim<-xidep[,2]
  ka<-psi[id,1]
  V<-psi[id,2]
  CL<-psi[id,3]
  ke<-CL/V
  ypred<-dose*ka/(V*(ka-ke))*(exp(-ke*tim)-exp(-ka*tim))
  return(ypred)
}

saemix.model<-saemixModel(model=model1cpt,
  description="One-compartment model with first-order absorption",
  # psi0=matrix(c(1.,20,0.5,0.1,0,-0.01),ncol=3, byrow=TRUE,dimnames=list(NULL, c("ka","V","CL"))),
  psi0=matrix(c(1.,20,0.5),ncol=3, byrow=TRUE,dimnames=list(NULL, c("ka","V","CL"))),
  transform.par=c(1,1,1),
  # covariate.model=matrix(c(0,0,1,0,0,0),ncol=3,byrow=TRUE),fixed.estim=c(1,1,1),
  covariance.model=matrix(c(1,0,0,0,1,0,0,0,1),ncol=3,byrow=TRUE),
  omega.init=matrix(c(1,0,0,0,1,0,0,0,1),ncol=3,byrow=TRUE),
  error.model="constant"
)

saemix.fit<-saemix(saemix.model,saemix.data,list(seed=632545,nb.chains=5, nbiter.saemix = c(300, 150)))
