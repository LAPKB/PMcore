library(here)
library(tidyverse)

i_am("transform_dataset.r")

data <- read.csv(here("datatmp.csv"))

data$input[data$input == 1] <- 0
data$outeq[data$outeq == 1] <- 0
data$id <- substr(data$id, 3, nchar(data$id))

data$X <- NULL

write.csv(data, here("data_pmcore.csv"), row.names = FALSE)


saemix_data <- data %>% 
  select(-c(c0, c1, c2, c3, wtmg, cfu0, cens, outeq, evid, dur, addl, ii, input)) %>%
  relocate(id, dose, time, out) %>%
  rename(
    Id = id,
    Dose = dose,
    Time = time,
    Concentration = out
  ) %>%
  mutate(Dose = first(Dose)) %>%
  filter(Concentration != ".")

write.csv(saemix_data, here("data_saemix.csv"), row.names = FALSE, na = ".")
