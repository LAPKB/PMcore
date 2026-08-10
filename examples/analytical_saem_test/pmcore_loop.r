library(rextendr)
library(here)

i_am("pmcore_loop.r")

# rust_source(here("main.rs"), 
#   dependencies = list(
#     anyhow = "1.0.100",
#     "extendr-api" = "0.9.0",
#     pmcore = "{ git = \"https://github.com/LAPKB/PMcore\", branch = \"SAEM_validation_theophylline\" }"
#   )
# )

# clean target folder
unlink(here("pmcore_output"), recursive = TRUE)


# load trial initial states
inits <- read.csv(here("random_init.csv"))

trial_loop <- function(ka, ke, v, trial_id) {
  system(paste(
    here("..", "..", "target", "debug", "examples", "analytical_saem_test"),
    as.character(ka),
    as.character(ke),
    as.character(v),
    here("converted_data_theo.csv"),
    here("pmcore_output", "run_data")
  ))

  pmcore_stats <- read.csv(here("pmcore_output", "run_data", "statistics.csv"))
  pmcore_trace <- data.frame(matrix(NA, nrow = max(pmcore_stats$cycle), ncol = 0))
  pmcore_trace$ka <- dplyr::filter(pmcore_stats, name == "ka", kind == "theta")$value
  pmcore_trace$V <- dplyr::filter(pmcore_stats, name == "v", kind == "theta")$value
  pmcore_trace$ke <- dplyr::filter(pmcore_stats, name == "ke", kind == "theta")$value

  stopifnot(all(c("ka", "V", "ke") %in% names(pmcore_trace)))

  pmcore_trace$cycle <- 1:nrow(pmcore_trace)
  pmcore_trace$trial_id <- trial_id
  col_names = FALSE
  if (trial_id == 0) {col_names = TRUE}

  write.table(pmcore_trace, 
              file = here("pmcore_output", "pmcore_trace.csv"), 
              append = TRUE, 
              sep = ",", 
              col.names = col_names, 
              row.names = FALSE)
}

for (i in 1:nrow(inits)) {
  row <- inits[i, ]
  trial_loop(row$ka, row$ke, row$v, row$trial_id)
}
