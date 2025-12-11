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
setwd(root.dir)

# Archive stuff (as a precaution)
system(paste("cp ", root.dir, "/../main.rs ", root.dir, sep = ''))
r.name = "out_sde_paper_sigma_eq_1pt0"
save_dir = paste(root.dir,"/../",r.name,sep='')
system(paste("mkdir ", save_dir, sep = ''))
system(paste("cp -r ../output/ ", save_dir))
system(paste("cp -r ../data/ ", save_dir))

# There are seven experiments, 0..6
# ke, ODE
# ke, SDE s1=0
# ke, SDE s1
# ke,v SDE s1=0, s2=0
# ke,v SDE s1, s2=0
# ke,v SDE s1, s2 ------
# ke,v ODE -------------

{
  {
    run_name.ode <- "SDE Paper\nODE(Ke,V; s=mu/1); s_obs = 1.0 + 0.2Y"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment6/'
    setwd(outdir); op.ode <- read.csv("op.csv"); theta.ode <- read_csv("theta.csv")
  } # load op.ode and theta.ode -- check labels
  {
    run_name.sde <- "SDE Paper\nSDE(Ke,V; s=mu/1); s_obs = 1.0 + 0.2Y"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment5/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # load op.sde and theta.sde
}
setwd(root.dir)

# outids <- unique(op$id[which(op$p    )])

# choose one to plot:
run_name <- run_name.ode
op <- op.ode # %>% filter(!id %in% outids) # c("id23"))
run_name <- run_name.sde
op <- op.sde # %>% filter(!id %in% outids) # c("id23"))

{
  against <- op$post_mean
  {
    plot.lim = c(0,max(max(op$obs),max(against)))
    plot(against, op$obs, type = "p", ylim = plot.lim, xlim = plot.lim,
       , xlab = "posterior expected concentration", ylab = "observed concentration"
       , main = paste("post_mean\n", run_name))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
  } # ... and plot obs vs. against
  {
    new_subj <- ""
    iii <- which(op$id %in% new_subj)
    lines(against[iii], op$obs[iii], col = "red", type = "p", pch = 16)
  } # OPTIONAL: overlay new_subj in "red"
  {
    BLQ = -1.0
    {
      op.blq <- op %>% filter(obs > BLQ)
      against.blq <- op.blq$post_mean
    } # op.blq <- remove BLQ observations
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
  against <- op$pop_mean # (op %>% filter(obs > 4))$pop_mean
  obs <- op$obs # (op%>% filter(obs > 4))$obs 
  # against <- log10(against)
  # obs <- log10(obs)
  {
    plot.lim = c(0,max(max(obs),max(against)))
    plot(against, obs, type = "p", ylim = plot.lim, xlim = plot.lim
       , xlab = "population expected concentration", ylab = "observed concentration"
       , main = paste("pop_mean\n",run_name)) # , pch = as.character(op$id))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
  } # obs vs. pred
  {
    iii <- which(op$id %in% new_subj)
    lines(against[iii], obs[iii], col = "red", type = "p", pch = 16)
  } # iii <- new_data or outliers
  {
    BLQ = -2
    op.blq <- op %>%
      filter(obs == BLQ) %>%
      # sigma is from main.rs
      mutate(obs = if_else(post_mean < BLQ + 1.93*(0.250 + 0.0625*obs), post_mean, obs))
    lines(op.blq$pop_mean, op.blq$obs, col = "red", type = "p") 
    abline(h = c(BLQ), col = "grey")
  } # if obs <=  BLQ: post -> pop, to visulize "estimable" BLQ predictions
  {
    summary(lm(obs ~ against))
    inter = lm(obs ~ against)[1]$coefficients[1]
    sl = lm(obs ~ against)[1]$coefficients[2]
    lines(plot.lim,
        inter + sl*plot.lim
        , lty = 7, lwd = 3, col = "grey"
    )
    # abline(h = c(BLQ), col = "grey")
  } # summary
}




