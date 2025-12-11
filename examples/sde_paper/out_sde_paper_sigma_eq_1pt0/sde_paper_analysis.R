#
# ~/src/lapk/PMcore/examples/sde_paper/output/sde_paper_analysis.R
#
# Rust code: ~/src/lapk/PMcore/examples/sde_paper/main.rs
#
#
#
library(tidyverse)
old.dir <- getwd()
root.dir <- "~/src/lapk/PMcore/examples/sde_paper/output"

# Archive stuff (as a precaution)
system(paste("cp ", root.dir, "/../main.rs ", root.dir, sep = ''))
r.name = "out_sde_paper_sigma_eq_1pt0"
save_dir = paste(root.dir,r.name,sep='')
