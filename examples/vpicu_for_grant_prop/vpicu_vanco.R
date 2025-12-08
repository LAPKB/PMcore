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
  # subj 46 has a dose of 1 at time 0; suspect this is supposed to be 1000 -- manually corrected
  write_csv(ndata_csv,"/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/vpicu.csv", na = ".")

}# transform original csv as necessary and write csv

{
  {
    # this reduces the dataset to 1 line for each subject, with it's #of lines
    # use it (in concert w/printing the subj, support, like from pharmsol) to
    # see what subject might be hanging the program
    d.out <- read_csv("/Users/wyamada/src/lapk/PMcore/file.txt") %>%
      count(id,sort=TRUE)
  } # found number of lines in each group
  {
    op[which(op$obs > 8 & op$post_mean < 5), ]
  } # looking for outliers in op
  
} # some data debugging stuff


{
  op <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode/op.csv")
  BLQ = 4.0;
  run_name <- "vpicu ODE"
  {
    op.blq <- op %>% filter(obs > BLQ)
    against.blq <- op.blq$post_mean
  } # op.blq <- remove BLQ observations
  against <- op$post_mean
  {
    plot.lim = c(0,max(max(op$obs),max(against)))
    plot(against, op$obs, type = "p", ylim = plot.lim, xlim = plot.lim,
         , xlab = "posterior expected concentration", ylab = "observed concentration"
         , main = paste("post_mean\n", run_name))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
  } # ... and plot obs vs. against
  {
    summary(lm(op.blq$obs ~ against.blq))
    {
      inter = lm(op.blq$obs ~ against.blq)[1]$coefficients[1]
      sl = lm(op.blq$obs ~ against.blq)[1]$coefficients[2]
      lines(plot.lim,
            inter + sl*plot.lim
            , lty = 7, lwd = 3, col = "grey"
      )
    } # plot reg line w/out BLQ
    # abline(h = c(BLQ), col = "grey")
  } # summary and regression line w/BLQ removed!!!
  # ------------------------------------------------------- POPULATION FITS ---
  against <- op.blq$pop_mean # (op %>% filter(obs > 4))$pop_mean
  obs <- op.blq$obs # (op%>% filter(obs > 4))$obs 
  # against <- log10(against)
  # obs <- log10(obs)
  {
    plot.lim = c(0,max(max(obs),max(against)))
    plot(against, obs, type = "p", ylim = plot.lim, xlim = plot.lim
         , xlab = "population expected concentration", ylab = "observed concentration"
         , main = paste("pop_mean\n",run_name)) # , pch = as.character(op$id))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
  } # obs vs. pred
  # ---
  {
    op.blq <- op %>%
      filter(obs == BLQ) %>% # find BLQ measurements
      # sigma is from main.rs
      mutate(obs = if_else(post_mean < BLQ + 1.93*(0.1 + 0.15*obs), post_mean, obs))
    lines(op.blq$pop_mean, op.blq$obs, col = "red", type = "p") 
    abline(h = c(BLQ), col = "grey")
  } # for (obs ==  BLQ) if CI95%(BLQ) covers the posterior pred, then use post instead of obs to visualize data
  # ---
  {
    summary(lm(obs ~ against))
    inter = lm(obs ~ against)[1]$coefficients[1]
    sl = lm(obs ~ against)[1]$coefficients[2]
    lines(plot.lim,
          inter + sl*plot.lim
          , lty = 7, lwd = 3, col = "grey"
    )
    # abline(h = c(BLQ), col = "grey")
  }
  
} # op plot
{
  # run wtCov.R
  theta_ode <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode/theta.csv")
  summary(theta_ode)
  np.stats(theta_ode)
} # inspect theta

setwd(goback)
