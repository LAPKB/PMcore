library(saemix)
library(ggplot2)
library(gridExtra)
library(tidyverse)
library(here)

i_am("compare_data.r")

pmcore_stats <- read.csv(here("pmcore_output", "pmcore_trace.csv"))
pmcore_stats <- split(pmcore_stats, pmcore_stats$trial_id)

saemix_stats <- read.csv(here("SAEMix_output", "saemix_trace.csv"))
saemix_stats <- split(saemix_stats, saemix_stats$trial_id)


ka_difs <- c()
v_difs <- c()
ke_difs <- c()
ke_v_difs <- c()

final_kas <- c()
final_vs <- c()
final_kes <- c()


for (i in 1:(length(pmcore_stats))) {
  ka_pmcore <- filter(pmcore_stats[[i]], name == "ka")[, c("cycle", "value")]
  ka_saemix <- saemix_stats[[i]][, c("cycle", "ka")]
  ka_dif <- array(ka_pmcore$value) - array(ka_saemix$ka)

  ka_difs[[length(ka_difs) + 1]] <- ka_dif

  final_kas[[length(final_kas) + 1]] <- (ka_pmcore[[length(ka_pmcore)]] + ka_saemix[[length(ka_saemix)]])/2


  v_pmcore <- filter(pmcore_stats[[i]], name == "v")[, c("cycle", "value")]
  v_saemix <- saemix_stats[[i]][, c("cycle", "V")]
  v_dif <- array(v_pmcore$value) - array(v_saemix$V)

  v_difs[[length(v_difs) + 1]] <- v_dif

  final_vs[[length(final_vs) + 1]] <- (v_pmcore[[length(v_pmcore)]] + v_saemix[[length(v_saemix)]])/2


  ke_pmcore <- filter(pmcore_stats[[i]], name == "ke")[, c("cycle", "value")]
  ke_saemix <- saemix_stats[[i]][, c("cycle", "ke")]
  ke_dif <- array(ke_pmcore$value) - array(ke_saemix$ke)

  ke_difs[[length(ke_difs) + 1]] <- ke_dif

  final_kes[[length(final_kes) + 1]] <- (ke_pmcore[[length(ke_pmcore)]] + ke_saemix[[length(ke_saemix)]])/2


  ke_v_pmcore <- array(ke_pmcore$value)/array(v_pmcore$value)
  ke_v_saemix <- array(ke_saemix$ke)/array(v_saemix$V)
  ka_v_dif <- array(ke_v_pmcore) - array(ke_v_saemix)
  ke_v_difs[[length(ke_v_difs) + 1]] <- ka_v_dif
}

# average final values
final_average <- c(
  final_ka = mean(unlist(final_kas)),
  final_v = mean(unlist(final_vs)),
  final_ke = mean(unlist(final_kes)),
  final_ke_v = mean(unlist(final_kes)) / mean(unlist(final_vs))
)


# Element-wise average
ka_difs_average <- Reduce("+", ka_difs) / length(ka_difs) / final_average[[1]]
v_difs_average <- Reduce("+", v_difs) / length(v_difs) / final_average[[2]]
ke_difs_average <- Reduce("+", ke_difs) / length(ke_difs) / final_average[[3]]
ke_v_difs_average <- Reduce("+", ke_v_difs) / length(ke_v_difs)


# , ylim=c(-.1, .1)
par(pty = "s")
plot(ka_difs_average, main="ka difference", xlab="cycle", ylab="ka", col="blue", pch=19, panel.first = grid(nx = NULL, ny = NULL), ylim=c(-1, 1)) # -final_average[[1]], final_average[[1]]
plot(v_difs_average, main="v difference", xlab="cycle", ylab="v", col="blue", pch=19, panel.first = grid(nx = NULL, ny = NULL), ylim=c(-1, 1))
plot(ke_difs_average, main="ke difference", xlab="cycle", ylab="ke", col="blue", pch=19, panel.first = grid(nx = NULL, ny = NULL), ylim=c(-1, 1))
plot(ke_v_difs_average, main="ke/v difference", xlab="cycle", ylab="ke/v", col="blue", pch=19, panel.first = grid(nx = NULL, ny = NULL))
