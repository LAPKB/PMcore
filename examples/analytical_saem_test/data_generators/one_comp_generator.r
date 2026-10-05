library(Pmetrics)

# simulate a known model
# pass observations to Romain to model in monolix and Joshua to model in pmetrics

setwd("~/Documents/lapk/saem_validation")

#
# One compartment
# w/no error of any sort
# and a lot of samples
# (error can be added to the completed simulation before optimization)
#
mod <- PM_model$new(  pri = list(
    ka = ab(0.0, 5.000),
    ke = ab(0.0, 5.000),
    v = ab(0.0, 100.000)
  ),
  eqn = function () 
  {
    dX[1] <- B[1] - ka * x[1]
    dX[2] <- ka * x[1] - ke * x[2]
  },
  out = function () 
  {
    y[1] <- x[2]/v
  },
  err = list(
    proportional(0.2, c(0.1, 0.1, 0.0, 0.0), outeq=1)
  )
)

# data description
pop001 <- list(
  wt = 1,
  mean = list(ka = 0.8, ke = 0.18, v = 63.0),
  cov = diag(c(0.0064, 0.000324, 39.69)) # s = 10%
)
template <- PM_data$new(data = "sim001.csv")

# debugonce(PM_sim$new)
# simulate
sim001 <- PM_sim$new( # PM_result$sim( # 
  poppar = pop001, limits = NA,
  data = template, model = mod,
  include = c(5), # more than 1 subject leads to an R error
  nsim = 64, # nsim=101 => ! Could not evaluate cli `{}` expression: `nsub`.
  predInt = c(0, 64, 0.5),
  makecsv = "saem_valid002_32_one_comp_random_oversampled"
)

plot(sim001,log=F)

#
for (x in 1:32) {
  if (x > 9) {
    d.file <- paste("saem_valid002_",x,"_one_comp_random_oversampled.csv",sep='')
    d.file.parsed <- paste("saem_valid002_",x,".csv",sep='')
  } else { # assume these are x in 1:9
    d.file <- paste("saem_valid002_0",x,"_one_comp_random_oversampled.csv",sep='')
    d.file.parsed <- paste("saem_valid002_0",x,".csv",sep='')
  }

# NPAG optimization; parse dense output first.
newdata <- read.csv(d.file)
id_list <- unique(newdata$ID) # this should be aligned w/sim001$parValues
newdata <- read.csv(d.file) %>%
  # filter(TIME %in% c(0,7.5,8,9.0,15.5,16,17.0,23.5,24,25.0,31.5,32,33.0,39.5,40,41.0,47.5,48,49.0,55.5,56)) %>%
  group_by(ID) %>%
    filter(TIME %in% c(0,sample(seq(0.5,7.5,0.5),size=1),
       8,sample(seq( 8.5,15.5,0.5),size=1),
      16,sample(seq(16.5,23.5,0.5),size=1),
      24,sample(seq(24.5,31.5,0.5),size=1),
      32,sample(seq(32.5,39.5,0.5),size=1),
      40,sample(seq(40.5,64.0,0.5),size=1))
    ) %>% # mutate(obs_no = id_list[which(id_list == ID)]) %>%
    mutate(ka = sim001$parValues$ka[which(id_list == ID)]) %>%
    mutate(ke = sim001$parValues$ke[which(id_list == ID)]) %>%
    mutate(v = sim001$parValues$v[which(id_list == ID)]) %>%
  ungroup() %>%
  rename_with(tolower) %>%
  mutate(nonoise=as.numeric(out)) %>%
  mutate(obserr=(nonoise/20)^2) %>%
  mutate(obserr=rnorm(n(),0,obserr)) %>%
  mutate(out=nonoise+obserr) %>%
  mutate(out=if_else(out>0,out,0,missing=NA)) %>%
  mutate(nonoise=if_else(time==0,0,nonoise)) %>%
  mutate(obserr=if_else(time==0,0,obserr)) %>%
  mutate(outeq=if_else(outeq==1,0,NA)) %>% # not sure why these have to be recast to 0 ... as Joshua
  mutate(input=if_else(input==1,0,NA)) %>%
  filter_out(time %in% c(0,8,16,24,32,40,48,56) & out > 0)
# newdata$out[which(newdata$out < 0)]
write.csv(newdata,file=d.file.parsed, na='.')
}
  # optimize
simdata <- PM_data$new(d.file.parsed)
run1 <- mod$fit(data = simdata, run = 1, overwrite = TRUE, path = "Runs", cycles=5000)

# write.csv(sim001$parValues, file="parValues.csv") # is integrated in the newdata dataframe
