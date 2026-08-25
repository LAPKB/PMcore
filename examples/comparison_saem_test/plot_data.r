library(saemix)

par(pty = "s")

s.pmcore <- read.csv("outputs/pmcore_output/predictions.csv")
y <- s.pmcore$observation;
x <- s.pmcore$conditional_prediction
plot(x,y, ylab = "Observation", xlab = "Conditional Prediction", xlim = c(0, 140), ylim = c(0, 140))
curve(x+0, from = 0, to = 140, col = "blue", lwd = 2, add = TRUE)
grid()
y <- s.pmcore$observation;
x <- s.pmcore$population_prediction
plot(x,y, ylab = "Observation", xlab = "Population Prediction", xlim = c(0, 140), ylim = c(0, 140))
curve(x+0, from = 0, to = 140, col = "blue", lwd = 2, add = TRUE)
grid()

plot(saemix.fit, plot.type="observations.vs.predictions")


ka <- saemix.fit@results@fixed.psi[1]
ke <- saemix.fit@results@fixed.psi[2]
V  <- saemix.fit@results@fixed.psi[3]
