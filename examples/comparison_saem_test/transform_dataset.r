library(here)
library(tidyverse)

i_am("transform_dataset.r")

data <- read.csv(here("saem_valid001_one_comp_no_error_oversampled.csv"))

data$INPUT[data$INPUT == 1] <- 0
data$OUTEQ[data$OUTEQ == 1] <- 0
data$ID <- substr(data$ID, 3, nchar(data$ID))

write.csv(data, here("data_pmcore.csv"), row.names = FALSE)


saemix_data <- data %>% 
  select(-c(C0, C1, C2, C3, WTMG, CFU0, CENS, OUTEQ, EVID, DUR, ADDL, II, INPUT)) %>%
  relocate(ID, DOSE, TIME, OUT) %>%
  rename(
    Id = ID,
    Dose = DOSE,
    Time = TIME,
    Concentration = OUT
  ) %>%
  mutate(Dose = first(Dose)) %>%
  filter(Concentration != ".")

write.csv(saemix_data, here("data_saemix.csv"), row.names = FALSE, na = ".")
