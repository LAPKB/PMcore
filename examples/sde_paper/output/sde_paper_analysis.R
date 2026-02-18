#
# ~/src/lapk/PMcore/examples/sde_paper/output/sde_paper_analysis.R
#
# Rust code: ~/src/lapk/PMcore/examples/sde_paper/main.rs
#
# wmy20260120 I don't remember what I was doing. I _think_ this is the correct file to
#.  figure things out:
#.  checkout wmy_npag_sde_01 in both PMcore and pharmsol, then answer:
#.  1) I recall deciding a centering function is not necessary (but phrmsol is looking at
#.     outeq < 2; not <1 ... so not sure ... but out[1] = out[0] ... so I have a square???
#.     YES.)
#.  2) sigma = f * mean; e.g. v0/5, f=0.2
#.     this is not what I remember .. and I made a new example 12 ... that does what I
#.     think is correct:
#.     if (GENDATA) {d[1] = s1 = CV_PERCENT*Ke0} else (d[1] = s1, the optimized r.v.)
#.  3) CLUE archive to ...sde_paper/output_1_80 which means simulate using 1 particle and
#.     optimize using 80 particles ... NO: we are using 1_47
#.  4) Conclusion: This is the correct Rscript to keep working.
#
library(tidyverse)
old.dir <- getwd()
root.dir <- "~/src/lapk/PMcore/examples/sde_paper/output"
setwd(root.dir)

# Archive the run (as a precaution -- do after analysis, too)
r.name = "out_N_eq_256_p_eq_1_128_fs1_eq_100_fs2_eq_10" # last set of graphs
r.name = "out_N_eq_256_p_eq_1_128_fs1_eq_10_fs2_eq_10_c0eq1c1eq0pt15" # 
r.name = "out_N_eq_256_p_eq_1_128_fs1_eq_1000_fs2_eq_1000_c0eq1c1eq0pt15" #
{
  gb.dir <- getwd();setwd(root.dir)
  system(paste("cp ", root.dir, "/../main.rs ", root.dir, sep = ''))
  save_dir = paste(root.dir,"/../",r.name,sep='')
  system(paste("mkdir ", save_dir, sep = ''))
  system(paste("cp -r ../output/ ", save_dir))
  system(paste("cp -r ../data/ ", save_dir))
  setwd(gb.dir); remove(gb.dir)
} # cp main.rs, data, and output to sde_example/r.name/

# There are nine experiments, 0..8 {20260120 we've added more}
# 0: ke   ODE
# 1: ke   SDE s1=0
# 2: ke   SDE s1
# 3: ke,v SDE s1=0, s2=0
# 4: ke,v SDE s1, s2=0
# 5: ke,v SDE s1, s2
# 6: ke,v ODE
# 7: v    ODE
# 8: v    SDE s2

{ # load one ode and one sde, then go back to the root.dir
  {
    run_name.ode <- "(0) ODE(2:Ke, s1=Ke0/10; V=1, s2=0)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment0/'
    setwd(outdir); op.ode <- read.csv("op.csv"); theta.ode <- read_csv("theta.csv")
  } # ode 0 : ke v=1 -- use data from exp 2
  {
    run_name.sde <- "(2) SDE(Ke, s1=Ke0/10; V=1, s2=0)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment2/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # sde 2 : ke v=1 s1 s2=0
  {
    run_name.sde <- "SDE Paper (1)\nSDE(Ke; V=1, s=0); s_obs = 1.0 + 0.2Y"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment1/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # sde 1 : ke v=1 s1=s2=0 // 1 and 0 should be near identical
  #
  {
    run_name.sde <- "SDE Paper (4)\nSDE(Ke,V; s1=Ke0/1, s2=0); s_obs = 1.0 + 0.2Y"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment4/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # sde 4 : ke v s1 s2=0 // one off test
  #
  {
    run_name.ode <- "(6) ODE(5:Ke0,V0,s1=Ke0/10,s2=v0/10)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment6/'
    setwd(outdir); op.ode <- read.csv("op.csv"); theta.ode <- read_csv("theta.csv")
  } # ode 6 : ke v -- use data from exp 5
  {
    run_name.sde <- "(5) SDE(Ke0,V0,s1=Ke0/10,s2=v0/10)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment5/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # sde 5 : ke v s1 s2
  {
    run_name.sde <- "SDE Paper (3)\nSDE(Ke,V; s=0)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment3/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # sde 3 : ke v s1=s2=0 // 3 and 6 should be near identical
  #
  {
    run_name.ode <- "(7) ODE(8:V0, s2=v0/10; Ke=1, s1=0)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment7/'
    setwd(outdir); op.ode <- read.csv("op.csv"); theta.ode <- read_csv("theta.csv")
  } # ode 7 : v ke=-1 -- use data from exp 2
  {
    run_name.sde <- "(8) SDE(v0, s2=v0/10; Ke0=1, s1=0)"
    outdir <- '~/src/lapk/PMcore/examples/sde_paper/output/experiment8/'
    setwd(outdir); op.sde <- read.csv("op.csv"); theta.sde <- read_csv("theta.csv")
  } # sde 8 : v s2; ke=-1 s1=0
} # load one ode and one sde
setwd(root.dir)

{
  # outids <- unique(op$id[which(op$p    )])
  summary(op$obs)
  summary(op$post_mean)
  summary(op$pop_median)
} # find outlier ids

# choose one to plot (one sde and one ode was loaded above)
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
  summary(lm(op$obs ~ against))
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
  summary(lm(op$obs ~ against))
} # plot op post and population mean

# --- THETA ------

#
s_fact = "(10, 10)"
{
  m1 = 0.5
  m2 = 1.5
  s1 = 0.05
  s2 = 0.15 # (m2 - (m1 + 3.0*s1))/3.0
  ke_range <- c(0.1, 3.0)
  v_range <- c(0.1, 3.0)
  init_cond.text <- paste("sobol(347); N=4097");
} # data parameters, e.g. ranges
{
  options(pillar.sigfig = 6)
  setwd("/Users/wyamada/src/lapk/PMcore/examples/sde_paper/data/")
  exper2 <- read_csv("experiment2.csv") # s1
  exper5 <- read_csv("experiment5.csv") # s1 and s2
  exper8 <- read_csv("experiment8.csv") # s1 and s2
  setwd("/Users/wyamada/src/lapk/PMcore/examples/sde_paper/")
} # load experiments 2, 5 and 8.csv
# note: for vanco I had two regular outputs and a running chi^2 
#
d.truth <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/sde_paper/data/population.csv")
matrix(c(mean(d.truth$k0), mean(d.truth$v0)
         ,var(d.truth$k0)^0.5,var(d.truth$v0)^0.5),byrow=T,nrow=2)
s_fact = "/(10,10)"
#
# !!! ARCHIVE DATA _before_ DOING ANALYSES !!! for example:
#
# mkdir ./examples/sde_paper/output_1_80; mkdir ./examples/sde_paper/output_1_80/output_s.eq.0.1
# cp -r ./examples/sde_paper/output/ ./examples/sde_paper/output_1_80/output_s.eq.0.1
#
# ------# Choose one of: S0S0, SO, StO, SS, StSt #-------#
#
#
{
  d.truth <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/sde_paper/data/population.csv")
  names(d.truth) <- c("index","ke0","v0")
  d.ode <- theta.ode
  d.sde <- theta.sde
} # d.truth
{
    names(d.ode) <- c("ke0", "prob")
    names(d.sde) <- c("ke0",   "v0",   "fs1",   "fs2",   "prob")
} # d.ode and d.sde for 2 vs 0 -- should be OK for ALL d.ode AND d.sde
{
    names(d.ode) <- c("ke0","v0","prob")
    names(d.sde) <- c("ke0","v0","fs1","fs2","prob")
} # d.ode and d.sde for 6 vs 5
{
  names(d.ode) <- c("ke0","v0","fs1","fs2" ,"prob")
  names(d.sde) <- c("ke0","v0","fs1","fs2" ,"prob")
} # d.ode and d.sde for 8 vs 7
# d.truth,ode,sde <- population, ODE, and SDE examples 5 and 6
#
exp.sigma <- paste("s=mu/",s_fact,sep='')
# (1)
ylimke = c(0,5) # ad hoc ... just depends where the data lay
ylimv = c(0,8)
{
  {
    col.ode <- "khaki3"; col.sde <- "indianred3"
    col.text <- paste("POP = grey/blue; ODE = ", col.ode, "; SDE = ", col.sde,sep = '')
    m.text <- paste(col.text, "\n", exp.sigma, sep = '')
  } # assign colors
  {
    bw.ke =   (max(d.truth$ke0) - min(d.truth$ke0))/(100/3); # 0.1865
    xlabke = paste("ke0(bw = (max-min of pop)/33); s_f = ",s_fact)
    bw.vol = (max(d.truth$v0) - min(d.truth$v0))/(100/3); # 0.07139
    xlabv = paste("v0(bw = (max-min of pop)/33); s_f = ", s_fact)
    # main.txt <- paste("POP = grey/blue; ODE = ", col.ode, "; SDE = ", col.sde, "\n s = ", exp.sigma, sep = '')
    main.txt <- m.text
    xlimke <- c(-0.1, m2 + 5 * ske) # search space limit
    if (xlimke[2] < max(c(d.sde$ke0,d.ode$ke0))) {
      xlimke[2] = 1.2*max(c(d.sde$ke0,d.ode$ke0))
    }
    xlimv = c(-0.1, 2)
  } # graph parameters, e.g. limits, bandwidths, xlabel
  # --- don't touch below this. -- unless you need to adjust plot defaults
  {
    par( mfrow= c( 1, 2 ) )
    # plot(density(d.truth$k0), bw = 0.001 # bw.exp / 50
    hist(d.truth$ke0, breaks = 51, freq = F
         , xlim = xlimke, ylim = ylimke
         , lty = 0 # 6 # 0 for histogram
         # , lwd = 5
         , col = "lightgrey", xlab = xlabke, main = "")
    # , main = main.txt)
    lines(density(d.truth$ke0,bw = bw.ke), xlim = xlimke, lty = 7, col = "steelblue2", lwd = 4)
    lines(density(d.ode$ke0, bw = bw.ke, weights = d.ode$prob), xlim = xlimke, lty = 1, col = col.ode, lwd = 3)
    lines(density(d.sde$ke0, bw = bw.ke, weights = d.sde$prob), xlim = xlimke, lty = 1, col = col.sde, lwd = 2)
    hist(d.truth$v0, breaks = 51, freq = F
         , xlim = xlimv, ylim = ylimv
         , lty = 0, col = "lightgrey", xlab = xlabv, main = "" )
    # , main = main.txt)
    lines(density(d.truth$v0, bw = bw.vol)
          , xlim = xlimv, ylim = ylimv
          , lty = 7, lwd = 4, col = "steelblue2")
    lines(density(d.ode$v0, bw = bw.vol, weights = d.ode$prob)
          , lty = 1, col = col.ode, lwd = 3)
    lines(density(d.sde$v0, bw = bw.vol, weights = d.sde$prob)
          , lty = 1, col = col.sde, lwd = 2)
    mtext(main.txt, line = 0, side = 3, mar = -2, outer = TRUE)
  } # graphs
} # sde_paper graphs (limits are inside)
# (2)
summary(d.truth[,2:3])
c(var(d.truth[,2])^0.5,var(d.truth[,3])^0.5)
np.stats(d.ode)
np.stats(d.sde)
# (3) Wasserstein distance from population; c(ODE,SDE)
{
  # https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/
  # package `transport'
  # wasserstein(a, b, p=1, tplan=NULL, costm=NULL, prob=TRUE, ...)
  N = length(d.truth$v)
  a <- wpp(d.truth[,2:3], rep(1/N,N))
  b <- wpp(d.ode[,1:2], t(d.ode[,3]))
  c <- wpp(d.sde[,1:2], t(d.sde[,length(d.sde)])) # 5 (3 if no sigma optimization)
} # a <- population density; b <- ode; c <- sde
c(wasserstein(b,a,p=1,prob=TRUE), wasserstein(c,a,p=1,prob=TRUE)) # * 2^0.5
# Wasserstein distance from uniform (normalized distributions)
{
  vmin = min(v_range); vmax = max(v_range);
  kemin = min(ke_range); kemax = max(ke_range);
  dV = (vmax - vmin)/N
  xV = seq(1,N) * dV + (vmin - dV/2)
  xV <- (xV - vmin)/(vmax - vmin)
  dKe = (kemax - kemin)/N
  xKe = seq(1,N) * dKe + (kemin - dKe/2)
  xKe <- (xKe - kemin)/(kemax - kemin) # Ke should now be on [0,1]
  dflat <- as_tibble(matrix(c(xKe,xV), nrow = N, byrow = F))
  a <- wpp(dflat[,], rep(1/N,N))
  # now need to normalize the Ke and V distributions
  d.sde$ke0 <- (d.sde$ke0 - kemin)/(kemax - kemin)
  d.sde$v0 <- (d.sde$v0 - vmin)/(vmax - vmin)
  d.ode$ke0 <- (d.ode$ke0 - kemin)/(kemax - kemin)
  d.ode$v0 <- (d.ode$v0 - vmin)/(vmax - vmin)
  b <- wpp(d.ode[,1:2], t(d.ode[,3]))
  c <- wpp(d.sde[,1:2], t(d.sde[,length(d.sde)]))
} # a <- uniform over optimized range: c(ODE,SDE)
c(wasserstein(b,a,p=1,prob=TRUE), wasserstein(c,a,p=1,prob=TRUE)) / sqrt(2)
# (4) ... do you need regression stats?
runno <- "experiment0"
run_name <- "(1-47) OO" # \n t_obs in (0, 2pm0.5, 5pm0.5, 8pm0.5, 1)"
{
  runhome <- "/Users/wyamada/src/lapk/PMcore/examples/sde_paper/output/"
  op <- read.csv(paste(runhome,runno, "/op.csv",sep='')) # %>% filter(!id %in% outids) # c("id23"))
  #
  against <- op$pop_mean
  {
    {
      plot.lim = c(0,max(max(op$obs),max(against)))
      plot(against, op$obs, type = "p", ylim = plot.lim, xlim = plot.lim,
           , xlab = "pop expected concentration", ylab = "observed concentration"
           , main = paste("pop_mean\n", run_name))
      lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
    } # ... and plot obs vs. against
    # --- 
    BLQ = -1.0
    {
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
    }
  }
  summary(lm(op.blq$obs ~ against.blq))
  against <- op$post_mean
  {
    {
      plot.lim = c(0,max(max(op$obs),max(against)))
      plot(against, op$obs, type = "p", ylim = plot.lim, xlim = plot.lim,
           , xlab = "post expected concentration", ylab = "observed concentration"
           , main = paste("post_mean\n", run_name))
      lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
    } # ... and plot obs vs. against
    # --- 
    BLQ = -1.0
    {
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
    }
  }
  summary(lm(op.blq$obs ~ against.blq))
}
# (5) if you didn't do it above, then archive the directory, e.g.
# $cp -r ./examples/sde_paper/output/experiment<> ...

#
#. Temporary stuff for lab meeting ... hopefully 20260120
#
theta.sde <- read.csv("experiment12/theta.csv")
post.sde <- read_csv("experiment12/posterior.csv") %>%
  filter(prob > 10e-2)
