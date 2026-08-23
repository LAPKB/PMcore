library(saemix)

s.pmcore <- read.csv("outputs/pmcore_output/predictions.csv")
x <- s.pmcore$observation;
y <- s.pmcore$population_prediction
plot(x,y)
x <- s.pmcore$observation;
y <- s.pmcore$conditional_prediction
plot(x,y)

plot(saemix.fit, plot.type="observations.vs.predictions")


dose <- 2000
tau <- 8
ka <- saemix.fit@results@fixed.psi[1]
ke <- saemix.fit@results@fixed.psi[2]
V  <- saemix.fit@results@fixed.psi[3]


plot(s.pmcore$time, s.pmcore$observation)

curve(dose * ka / (V * (ka - ke)) * (exp(-ke * x)/(1 - exp(-ke * tau)) - exp(-ka * x)/(1 - exp(-ka * tau))), 
  col = "blue", lwd = 2, 
  main = "Plot of Function", xlab = "Time", ylab = "Concentration", add = TRUE)
