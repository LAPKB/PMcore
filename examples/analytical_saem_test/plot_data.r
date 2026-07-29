library(saemix)
library(ggplot2)
library(gridExtra)
library(tidyverse)


pmcore_data = read.csv("examples/analytical_saem_test/logs/pmcore_converted.csv")


# red -> pmcore
# blue -> saemix

# ka graph
max_len <- max(length(saemix.fit@results@allpar[, 1]), length(pmcore_data["Ka"]))
ka_results <- data.frame(
  c(saemix.fit@results@allpar[, 1], rep(NA, max_len - length(saemix.fit@results@allpar[, 1]))), 
  c(pmcore_data$Ka, rep(NA, max_len - length(pmcore_data$Ka)))
)
colnames(ka_results) <- c("saemix_ka", "pmcore_ka")
ka_results$iter <- seq_len(nrow(ka_results))

ggplot(ka_results, aes(x = iter)) +
  geom_line(aes(y = saemix_ka), color = "blue", size = 1) +
  geom_line(aes(y = pmcore_ka), color = "red", size = 1)



# v graph
max_len <- max(length(saemix.fit@results@allpar[, 2]), length(pmcore_data["V"]))
v_results <- data.frame(
  c(saemix.fit@results@allpar[, 2], rep(NA, max_len - length(saemix.fit@results@allpar[, 2]))), 
  c(pmcore_data$V, rep(NA, max_len - length(pmcore_data$V)))
)
colnames(v_results) <- c("saemix_v", "pmcore_v")
v_results$iter <- seq_len(nrow(v_results))

ggplot(v_results, aes(x = iter)) +
  geom_line(aes(y = saemix_v), color = "blue", size = 1) +
  geom_line(aes(y = pmcore_v), color = "red", size = 1)


# ke graph
max_len <- max(length(saemix.fit@results@allpar[, 3]), length(pmcore_data["Ke"]))
ke_results <- data.frame(
  c(saemix.fit@results@allpar[, 3], rep(NA, max_len - length(saemix.fit@results@allpar[, 3]))), 
  c(pmcore_data$Ke, rep(NA, max_len - length(pmcore_data$Ke)))
)
colnames(ke_results) <- c("saemix_cl", "pmcore_ke")
ke_results$saemix_ke <- ke_results$saemix_cl / v_results$saemix_v
ke_results$iter <- seq_len(nrow(ke_results))

ggplot(ke_results, aes(x = iter)) +
  geom_line(aes(y = saemix_ke), color = "blue", size = 1) +
  geom_line(aes(y = pmcore_ke), color = "red", size = 1)



plot(saemix.fit,plot.type="convergence")
# plot(saemix.fit,plot.type="likelihood")

# par(mfrow=c(3,4))
# saemix.plot.fits(saemix.fit,new=FALSE,ilist=1:12,smooth=TRUE,ylog=T,pch=1,
# col="Blue",xlab="Time in hr",ylab="Theophylline concentrations (mg/L)")


# # load rust data
# rust_output <- read.csv("examples/analytical_saem_test/pmcore_output/predictions.csv") %>%
#   filter(!is.na(obs))

# # rust_pop_parameters <- read.csv("examples/analytical_saem_test/output/population.csv")[2]$mu
# # rust_individual_parameters <- arrange(read.csv("examples/analytical_saem_test/output/individual_parameters.csv"), id)

# analytic_function <- function(t, ka, ke, v, x0) {
#   x0*ka/(v*(ka-ke))*(exp(-ke*t)-exp(-ka*t))
# }

# # population

# ggplot(rust_output, aes(time, pred_population)) + geom_point() + facet_wrap(~ id, 3, 4)

# # SAEMix data
# r_results <- cbind(saemix.fit@results@predictions, saemix.fit@data@data)
# ggplot(r_results, aes(Time, ppred)) + geom_point() + facet_wrap(~ Id, 3, 4)

# # population estimate vs actual
# ggtitle("Population Estimate vs Observed")
# pop_estim_plot <- ggplot(data, aes(Time, Concentration)) + geom_point() + facet_wrap(~ Id, 3, 4)
# for (i in 1:12) {
#   pop_estim_plot <- pop_estim_plot +
#     stat_function(
#       data = subset(data, Id == i),
#       fun = \(t) analytic_function(t, saemix.fit@results@fixed.psi[1], saemix.fit@results@fixed.psi[3]/saemix.fit@results@fixed.psi[2], saemix.fit@results@fixed.psi[2], data[i*10, 2]),
#       color = "blue", linewidth = 1
#     ) + 
#     stat_function(
#       data = subset(data, Id == i),
#       fun = \(t) analytic_function(t, rust_pop_parameters[1], rust_pop_parameters[2], rust_pop_parameters[3], data[i*10, 2]),
#       color = "red", linewidth = 1
#     )
# }
# pop_estim_plot

# # rust stem plot
# ggtitle("Deviation(Rust Population, Observed)")
# rust_output$stem <- (rust_output$obs - rust_output$pred_population)
# ggplot(rust_output, aes(time, stem)) + geom_point() + facet_wrap(~ id, 3, 4)

# # SAEMix stem plot
# ggtitle("Deviation(SAEMix Population, Observed)")
# r_results$stem <- (r_results$Concentration - r_results$ppred)
# ggplot(r_results, aes(Time, stem)) + geom_point() + facet_wrap(~ Id, 3, 4)

# # Rust vs SAEMix
# ggtitle("Deviation(Rust Population, SAEMix Population)")
# deviation_population <- data.frame(matrix(NA, 120, 0))
# deviation_population$deviation <- (rust_output$pred_population - r_results$ppred)
# deviation_population$time <- rust_output$time
# deviation_population$id <- rust_output$id
# ggplot(deviation_population, aes(time, deviation)) + geom_point() + facet_wrap(~ id, 3, 4)


# # individual

# # rust data
# rust_output <- read.csv("examples/analytical_saem_test/pmcore_output/predictions.csv") %>%
#   filter(!is.na(obs))
# ggplot(rust_output, aes(time, pred_individual)) + geom_point() + facet_wrap(~ id, 3, 4)

# # SAEMix data
# r_results <- cbind(saemix.fit@results@predictions, saemix.fit@data@data)
# ggplot(r_results, aes(Time, ipred)) + geom_point() + facet_wrap(~ Id, 3, 4)


# # # individual estimate vs actual
# # ggtitle("Individual Estimate vs Observed")
# # indiv_estim_plot <- ggplot(data, aes(Time, Concentration)) + geom_point() + facet_wrap(~ Id, 3, 4)
# # for (i in 1:12) {
# #   indiv_estim_plot <- indiv_estim_plot +
# #     stat_function(
# #       data = subset(data, Id == i),
# #       fun = \(t) analytic_function(t, saemix.fit@results@map.psi[i, 1], saemix.fit@results@map.psi[i, 3]/saemix.fit@results@map.psi[i, 2], saemix.fit@results@map.psi[i, 2], data[i*10, 2]),
# #       color = "blue", linewidth = 1
# #     ) + 
# #     stat_function(
# #       data = subset(data, Id == i),
# #       fun = \(t) analytic_function(t, rust_individual_parameters[i, 2], rust_individual_parameters[i, 3], rust_individual_parameters[i, 4], data[i*10, 2]),
# #       color = "red", linewidth = 1
# #     )
# # }
# # indiv_estim_plot

# # rust stem plot
# rust_output$stem <- (rust_output$obs - rust_output$pred_individual)
# ggplot(rust_output, aes(time, stem)) + geom_point() + facet_wrap(~ id, 3, 4)

# # SAEMix stem plot
# r_results$stem <- (r_results$Concentration - r_results$ipred)
# ggplot(r_results, aes(Time, stem)) + geom_point() + facet_wrap(~ Id, 3, 4)

# # Rust vs SAEMix
# ggtitle("Deviation(Rust Population, SAEMix Population)")
# deviation_population1 <- data.frame(matrix(NA, 120, 0))
# deviation_population1$deviation <- (rust_output$pred_individual - r_results$ipred)
# deviation_population1$time <- rust_output$time
# deviation_population1$id <- rust_output$id
# ggplot(deviation_population1, aes(time, deviation)) + geom_point() + facet_wrap(~ id, 3, 4)
