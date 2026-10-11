library(here)
library(dplyr)
library(readr)
library(ggplot2)

i_am("data_analysis.r")

true_data <- distinct(read_csv(here("test_data", "pmcore_data", "saem_valid002_01.csv"), col_select = c("id","ka", "ke", "v"), col_types = c("c", "d", "d", "d")))
pop_parameters <- c(mean(true_data$ka), mean(true_data$ke), mean(true_data$v))

trace <- read.csv(here("outputs", "pmcore_trace.csv"))
pop_estimates <- trace[trace$subject == "population", ]

pop_estimates$ka_dif <- NA
pop_estimates$ke_dif <- NA
pop_estimates$v_dif <- NA

for (i in seq_len(nrow(pop_estimates))) {
  pop_estimates[i, "ka_dif"] <- pop_parameters[1] - pop_estimates[i, "ka"]
  pop_estimates[i, "ke_dif"] <- pop_parameters[2] - pop_estimates[i, "ke"]
  pop_estimates[i, "v_dif"] <- pop_parameters[3] - pop_estimates[i, "v"]
}

pop_ka_dif_mean <- mean(pop_estimates$ka_dif, na.rm = TRUE)
pop_ka_dif_sd <- sd(pop_estimates$ka_dif, na.rm = TRUE)

pop_ke_dif_mean <- mean(pop_estimates$ke_dif, na.rm = TRUE)
pop_ke_dif_sd <- sd(pop_estimates$ke_dif, na.rm = TRUE)

pop_v_dif_mean <- mean(pop_estimates$v_dif, na.rm = TRUE)
pop_v_dif_sd <- sd(pop_estimates$v_dif, na.rm = TRUE)

ggplot(pop_estimates, aes(x = ka_dif)) +
  stat_function(fun = dnorm, args = list(mean = pop_ka_dif_mean, sd = pop_ka_dif_sd), 
                color = "blue") +
  labs(title = "Population Ka Difference Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()

ggplot(pop_estimates, aes(x = ke_dif)) +
  stat_function(fun = dnorm, args = list(mean = pop_ke_dif_mean, sd = pop_ke_dif_sd), 
                color = "blue") +
  labs(title = "Population Ke Difference Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()

ggplot(pop_estimates, aes(x = v_dif)) +
  stat_function(fun = dnorm, args = list(mean = pop_v_dif_mean, sd = pop_v_dif_sd), 
                color = "blue") +
  labs(title = "Population V Difference Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()


indiv_estimates <- trace[trace$subject != "population", ]
indiv_estimates$ka_dif <- NA
indiv_estimates$ke_dif <- NA
indiv_estimates$v_dif <- NA

for (i in seq_len(nrow(indiv_estimates))) {
  indiv_estimates[i, "ka_dif"] <- true_data$ka[true_data$id == indiv_estimates[i, "subject"]] - indiv_estimates[i, "ka"]
  indiv_estimates[i, "ke_dif"] <- true_data$ke[true_data$id == indiv_estimates[i, "subject"]] - indiv_estimates[i, "ke"]
  indiv_estimates[i, "v_dif"] <- true_data$v[true_data$id == indiv_estimates[i, "subject"]] - indiv_estimates[i, "v"]
}

indiv_ka_dif_mean <- mean(indiv_estimates$ka_dif, na.rm = TRUE)
indiv_ka_dif_sd <- sd(indiv_estimates$ka_dif, na.rm = TRUE)

indiv_ke_dif_mean <- mean(indiv_estimates$ke_dif, na.rm = TRUE)
indiv_ke_dif_sd <- sd(indiv_estimates$ke_dif, na.rm = TRUE)

indiv_v_dif_mean <- mean(indiv_estimates$v_dif, na.rm = TRUE)
indiv_v_dif_sd <- sd(indiv_estimates$v_dif, na.rm = TRUE)

ggplot(indiv_estimates, aes(x = ka_dif)) +
  stat_function(fun = dnorm, args = list(mean = indiv_ka_dif_mean, sd = indiv_ka_dif_sd), 
                color = "blue") +
  labs(title = "Individual Ka Difference Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()

ggplot(indiv_estimates, aes(x = ke_dif)) +
  stat_function(fun = dnorm, args = list(mean = indiv_ke_dif_mean, sd = indiv_ke_dif_sd), 
                color = "blue") +
  labs(title = "Individual Ke Difference Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()

ggplot(indiv_estimates, aes(x = v_dif)) +
  stat_function(fun = dnorm, args = list(mean = indiv_v_dif_mean, sd = indiv_v_dif_sd), 
                color = "blue") +
  labs(title = "Individual V Difference Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()
