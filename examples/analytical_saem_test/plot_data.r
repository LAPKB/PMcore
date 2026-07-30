library(saemix)
library(ggplot2)
library(gridExtra)
library(tidyverse)

saemix_trace <- as.data.frame(saemix.fit@results@allpar)[-1, , drop = FALSE]

pmcore_stats <- read.csv("examples/analytical_saem_test/pmcore_output/statistics.csv")
pmcore_trace <- data.frame(matrix(NA, nrow = max(pmcore_stats$cycle), ncol = 0))
pmcore_trace$ka <- filter(pmcore_stats, name == "ka", kind == "theta")$value
pmcore_trace$V <- filter(pmcore_stats, name == "v", kind == "theta")$value
pmcore_trace$ke <- filter(pmcore_stats, name == "ke", kind == "theta")$value

stopifnot(all(c("ka", "V", "ke") %in% names(pmcore_trace)))
# stopifnot(nrow(saemix_trace) == nrow(pmcore_trace))


# red -> pmcore
# blue -> saemix

# ka graph
max_len <- max(length(saemix_trace$ka), length(pmcore_trace$ka))
ka_results <- data.frame(
  c(saemix_trace$ka, rep(NA, max_len - length(saemix_trace$ka))), 
  c(pmcore_trace$ka, rep(NA, max_len - length(pmcore_trace$ka)))
)
colnames(ka_results) <- c("saemix_ka", "pmcore_ka")
ka_results$iter <- seq_len(nrow(ka_results))

ggplot(ka_results, aes(x = iter)) +
  geom_line(aes(y = saemix_ka), color = "blue", size = 1) +
  geom_line(aes(y = pmcore_ka), color = "red", size = 1)



# v graph
max_len <- max(length(saemix_trace$V), length(pmcore_trace$V))
v_results <- data.frame(
  c(saemix_trace$V, rep(NA, max_len - length(saemix_trace$V))), 
  c(pmcore_trace$V, rep(NA, max_len - length(pmcore_trace$V)))
)
colnames(v_results) <- c("saemix_v", "pmcore_v")
v_results$iter <- seq_len(nrow(v_results))

ggplot(v_results, aes(x = iter)) +
  geom_line(aes(y = saemix_v), color = "blue", size = 1) +
  geom_line(aes(y = pmcore_v), color = "red", size = 1)


# ke graph
max_len <- max(length(saemix_trace$ke), length(pmcore_trace$ke))
ke_results <- data.frame(
  c(saemix_trace$ke, rep(NA, max_len - length(saemix_trace$ke))), 
  c(pmcore_trace$ke, rep(NA, max_len - length(pmcore_trace$ke)))
)
colnames(ke_results) <- c("saemix_ke", "pmcore_ke")
ke_results$iter <- seq_len(nrow(ke_results))

ggplot(ke_results, aes(x = iter)) +
  geom_line(aes(y = saemix_ke), color = "blue", size = 1) +
  geom_line(aes(y = pmcore_ke), color = "red", size = 1)



plot(saemix.fit,plot.type="convergence")
