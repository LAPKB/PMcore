library(here)
library(dplyr)
library(readr)
library(ggplot2)

i_am("data_analysis.r")

data_file_list <- list.files(here("test_data", "pmcore_data"), pattern = "*.csv", full.names = TRUE)
combined_data <- read_csv(data_file_list, id = "dataset_file_name", n_max = 1, col_select = c("ka", "ke", "v"))
combined_data$dataset_file_name <- sub(".*test_data/pmcore_data/", "", combined_data$dataset_file_name)

trace <- read.csv(here("outputs", "pmcore_trace.csv")) 

trace$ka_dif <- NA
trace$ke_dif <- NA
trace$v_dif <- NA

for (i in seq_len(nrow(trace))) {
  trace[i, "ka_dif"] <- combined_data[combined_data$dataset_file_name == trace[i, "dataset_file_name"], "ka"] - trace[i, "ka"]
  trace[i, "ke_dif"] <- combined_data[combined_data$dataset_file_name == trace[i, "dataset_file_name"], "ke"] - trace[i, "ke"]
  trace[i, "v_dif"] <- combined_data[combined_data$dataset_file_name == trace[i, "dataset_file_name"], "v"] - trace[i, "v"]
}

ka_dif_mean <- mean(trace$ka_dif, na.rm = TRUE)
ka_dif_sd <- sd(trace$ka_dif, na.rm = TRUE)

ke_dif_mean <- mean(trace$ke_dif, na.rm = TRUE)
ke_dif_sd <- sd(trace$ke_dif, na.rm = TRUE)

v_dif_mean <- mean(trace$v_dif, na.rm = TRUE)
v_dif_sd <- sd(trace$v_dif, na.rm = TRUE)

ggplot(trace, aes(x = ka_dif)) +
#   # Plot empirical density histogram
#   geom_histogram(aes(y = ..density..), binwidth = 0.001, fill = "lightgray", color = "black") +
  # Overlay theoretical normal distribution curve
  stat_function(fun = dnorm, args = list(mean = ka_dif_mean, sd = ka_dif_sd), 
                color = "blue", size = 1) +
  labs(title = "Ka Dif Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()

ggplot(trace, aes(x = ke_dif)) +
#   # Plot empirical density histogram
#   geom_histogram(aes(y = ..density..), binwidth = 0.001, fill = "lightgray", color = "black") +
  # Overlay theoretical normal distribution curve
  stat_function(fun = dnorm, args = list(mean = ke_dif_mean, sd = ke_dif_sd), 
                color = "blue", size = 1) +
  labs(title = "Ke Dif Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()

ggplot(trace, aes(x = v_dif)) +
#   # Plot empirical density histogram
#   geom_histogram(aes(y = ..density..), binwidth = 0.1, fill = "lightgray", color = "black") +
  # Overlay theoretical normal distribution curve
  stat_function(fun = dnorm, args = list(mean = v_dif_mean, sd = v_dif_sd), 
                color = "blue", size = 1) +
  labs(title = "V Dif Normal Curve Over Column", x = "Values", y = "Density") +
  theme_minimal()
