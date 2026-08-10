library(tidyverse)
library(saemix)
library(here)
data(theo.saemix)

use_one <- TRUE

new.csv <- theo.saemix %>% 
  mutate("EVID" = 0) %>%
  mutate("DUR" = NA) %>%
  mutate("ADDL" = NA) %>%
  mutate("II" = NA) %>%
  mutate("INPUT" = NA) %>%
  mutate("OUTEQ" = 0) %>%
  mutate("C0" = NA) %>%
  mutate("C1" = NA) %>%
  mutate("C2" = NA) %>%
  mutate("C3" = NA) %>%
  group_by(Id) %>%
  group_modify(~ add_row(.x, .before = 0, EVID = 1, Time = 0, Dose = first(.$Dose), Weight = first(.$Weight), Sex = first(.$Sex), INPUT = 0, DUR = 0))

if (use_one) {
  new.csv <- dplyr::filter(new.csv, Id < 3)
}

new.csv$Dose[duplicated(new.csv$Id)] <- NA
new.csv <- select(new.csv, "Id", "EVID", "Time", "DUR", "Dose", "ADDL", "II", "INPUT", "Concentration", "OUTEQ", "C0", "C1", "C2", "C3", "Weight", "Sex")
new.csv <- rename(new.csv, ID = Id, TIME = Time, DOSE = Dose, OUT = Concentration)
write.csv(new.csv, here("converted_data_theo.csv"), row.names = FALSE, na = ".")
