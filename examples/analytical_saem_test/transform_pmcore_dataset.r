library(tidyverse)
library(saemix)
library(here)

i_am("transform_pmcore_dataset.r")

files_to_clear <- list.files(here("test_data", "saemix_data"), full.names = TRUE)
file.remove(files_to_clear)

files <- list.files(path=here("test_data", "pmcore_data"), pattern="*.csv", full.names=TRUE, recursive=FALSE)
lapply(files, function(file) {
  file_name <- sub(paste0(".*", "examples/analytical_saem_test/test_data/pmcore_data/") , "", file)
  converted_csv <- read.csv(file, na.strings = ".") %>%
    select(ID, DOSE, TIME, OUT) %>%
    rename(Id = ID, Dose = DOSE, Time = TIME, Concentration = OUT) %>%
    fill(Dose, .by = Id, .direction = "down") %>%
    slice(-1, .by = Id)
  write.csv(converted_csv, here("test_data", "saemix_data", file_name), row.names = FALSE, na = ".")
})
