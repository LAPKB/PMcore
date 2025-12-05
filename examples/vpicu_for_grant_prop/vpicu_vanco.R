#
# vanco_vpicu_v002
#
goback <- getwd()

{#
  load(tidyverse)
  ndata_csv <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/raw_data_files/vanco-vpicu-12-3-25.csv") %>%
    mutate(INPUT = if_else(INPUT=="none",0.0,as.numeric(INPUT))) %>%
    mutate(CENS = if_else(CENS=="none","-99.0",'.')) %>%
    select(ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,C0,C1,C2,C3,WT,CRCL) %>% # ,CENS,CR,HT,MALE,ABSTIME,ENCOUNTER) %>%
    filter(!ID==25) # this subj has 800 hr pause in treatment, needs to be two subjects
  write_csv(ndata_csv,"/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/vpicu.csv", na = ".")

}# transform original csv as necessary and write csv

{
  # this reduces the dataset to 1 line for each subject, with it's #of lines
  # use it (in concert w/printing the subj, support, like from pharmsol) to
  # see what subject might be hanging the program
  d.out <- read_csv("/Users/wyamada/src/lapk/PMcore/file.txt") %>%
    count(id,sort=TRUE)
} # some data debugging stuff

{
  # run wtCov.R
  theta_ode <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode/theta.csv")
  np.stats(theta_ode)
} # inspect theta


setwd(goback)
