#
# vanco_vpicu_v002
#
goback <- getwd()
{#
  library(tidyverse)
  {
  ndata_csv_old <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/raw_data_files/vanco-vpicu-12-3-25.csv") %>%
    mutate(INPUT = if_else(INPUT=="none",0.0,as.numeric(INPUT))) %>%
    mutate(CENS = if_else(CENS=="none","-99.0",'.')) %>%
    select(ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,C0,C1,C2,C3,WT,CRCL,ENCOUNTER) %>% # ,CENS,CR,HT,MALE,ABSTIME,ENCOUNTER) %>%
    filter(!ID==25) %>% # this subj has 800 hr pause in treatment, needs to be two subjects
    group_by(ID) %>%
      filter(TIME <= max(TIME[which(EVID==0)])) # remove trailing doses
  # Manually corrected the following in the raw csv file:
  # subj 46 has a dose of 1 at time 0; suspect this is supposed to be 1000 -- manually corrected
  # subj 49 has missing doses at t=42,48,54,60,...90
  # subj 62 has timeing mixup w/dose and obs (7.6 is a trough, not peak)
  #.   62,1,3242.666666333,1,600,.,.,1,.,.,.,.,.,.,25,176.999999367857
  #.   62,0,3243.000000333,.,.,.,.,.,7.599999905,1,.,.,.,.,25,176.999999367857 -> moved to t=3242.0
  # subj 7 has peak/trough mixup at t = 435.75
  #.   7	1	435.5	1	700	.	.	1	.	.
  #.   7	0	435.75	.	.	.	.	.	10.80000019	1 ... move to 435.25
  #.   7	1	439.5	1	700	.	.	1	.	.
  # subj 54 has trough peak mixup
  #.   1	17	1	90	.	.	1	.	.
  #.   0	17.05	.	.	.	.	.	8.5	1 ... move to 16.95
  #.   1	23.25	1	90	.	.	1	.	.
  # subject 49 has missing doses ... added some doses prior to trough obs at 89
  #.  1	36	1	100	.	.	1	.
  #.  0	89.0833333	.	.	.	.	.	13.10000038
  #.  1	96.3333333	1	100	.	.	1	.
  # subject 37 ... based on population model, I think this obs is 22, not 12
  #   37	1	25.333333337	1	125	.	.	1	.
  #.  37	0	32.499999997	.	.	.	.	.	12.10000038
  #.  37	1	33.916666667	1	125	.	.	1	.
  #.  id      time outeq block      obs  pop_mean pop_median post_mean post_median
  #.  37    32.5000     0     0 12.1000   5.32210    16.7341    20.1523     20.1523
  # Problems not corrected:
  # subj 37 has a strange measurement at t = 32.5, suspect patient took a dive (???)
  #.   Don't know how to fix b/c this might be real IOV or IOS
  # subj 48 crcl going from 73 to 1085 in 5 hours prior to the only observation at end point
  # subj 58 has long absence of data (what does SDE do w/that!?!)
  #.   58,1,41.5,1,100,.,.,1,5.699999809,1,.,.,.,.,8.180000305,142.484997862725
  #.   58,1,177.49999997,1,75,.,.,1,.,.,.,.,.,.,8.180000305,142.484997862725
  #   AND a big change in crcl before last measurement (and a change in dose w/no observation to indicate change is necessary)
  #.   58,1,231.49999997,1,100,.,.,1,.,.,.,.,.,.,8.180000305,142.484997862725
  #.   58,0,236.49999997,.,.,.,.,.,9.5,1,.,.,.,.,8.180000305,94.9899962004001
  #
  #
  } # vanco-vpicu-12-3-25.csv
  {
    ndata_csv <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/raw_data_files/vanco-vpicu-2-5-26(in).csv") %>%
    # ndata_csv <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/raw_data_files/vpicu_2_5_edited.csv") %>% 
      mutate(INPUT = if_else(INPUT=="none",0.0,as.numeric(INPUT))) %>%
      mutate(CENS = if_else(CENS=="none","-99.0",'.')) %>%
      # 1) old dataset hangs at 98%, find offending objservations, and why (?):
      # filter(ENCOUNTER %in% unique(ndata_csv_old$ENCOUNTER)) %>% # this is the old dataset (that "works")
      # filter(ENCOUNTER %in% unique(ndata_csv_old$ENCOUNTER)[27:37]) %>%
      filter(!ENCOUNTER==5205652082) %>% # filter(!ID==47) # above subsetting suggests this is problematic obs
      # 2) hanging at 98%, again ... find the new subjects that are at fault (of 19):
      # filter(!ENCOUNTER %in% unique(ndata_csv_old$ENCOUNTER)) %>% 
      # filter(ENCOUNTER %in% c(new_encounters[1:6],new_encounters[8:11],new_encounters[12:19])) %>% # made this list in console
      filter(!ENCOUNTER==5119935423) %>% # 7th new subjects hangs
      filter(!ENCOUNTER==5172512816) %>% # 11th new subject hangs  (ID 35 in edited dataset)  
      filter(!ENCOUNTER==5133458366) %>% # 28th in new subject set (data is weird, 200Hr pause, double dosing, missing dose at 480)
      select(ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ,C0,C1,C2,C3,WT,CRCL,ENCOUNTER) %>% # ,CENS,CR,HT,MALE,ABSTIME,ENCOUNTER) %>%
      group_by(ID) %>%
        filter(TIME <= max(TIME[which(EVID==0)])) %>% # remove trailing doses
        mutate(DOSE=if_else(ENCOUNTER == 5204605996 & TIME == 0.0, 1000.0, as.numeric(DOSE))) # the raw record was DOSE=1
        # Changes by hand:
        # if(ENCOUNTER == 5211708330) {
        #  group_modify( ~ add_row(.x,
        #      ID=49,EVID=1,TIME=40,DUR=1,DOSE=100,ADDL='.',II='.',INPUT=1,OUT='.',OUTEQ='.',C0='.',C1='.',C2='.',C3='.',WT=6.9,CRCL=152.81,ENCOUNTER=5211708330,
        #      .after=8)
        #  ) # missing doses at T=40,48,...90 (copied CRCL down; *** looks suspicious ***)
        # }
        # ENCOUNTER = 5204605996 (ID=46 in vpicu_csv_old)
        # ENCOUNTER = 5261750012 (ID=62 in vpicu_csv_old)
        # ENCOUNTER = 540821547 (ID = 7 in vpicu_csv_old)
        # ENCOUNTER = 5223781351 (ID = 54 in vpicu_csv_old)
        # ENCOUNTER = 5176316090 (ID = 37 in vpicu_csv_old)
        # ENCOUNTER =  (ID =  in vpicu_csv_old)
        # 5205652082 was major problematic!!! 
        # 5177125843 (ID=39) removed observation of 32mg/L b/c seems unreasonable AND can NOT fit
  } # vanco-vpicu-2-5-26(in).csv [three subjects not fittable]
  write_csv(ndata_csv,"/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/vpicu.csv", na = ".")
}# read original csv as necessary and write "vpicu.csv"
setwd(goback)

{
  {
    # this reduces the dataset to 1 line for each subject, with it's #of lines
    # use it (in concert w/printing the subj, support, like from pharmsol) to
    # see what subject might be hanging the program
    d.out <- read_csv("/Users/wyamada/src/lapk/PMcore/file.txt") %>%
      count(id,sort=TRUE)
  } # found number of lines in each group
  {
    print(n=26, op[which(op$obs > 15 & op$pop_mean < 5), ])
  } # looking for outliers in op
  
} # some data debugging snippets

{
  op <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode/op.csv")
  op <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde_sigma_only/op.csv")
  BLQ = 4.0;
  run_name <- "vpicu SDE sigma only"
  {
    op.blq <- op %>% filter(obs > BLQ)
    against.blq <- op.blq$post_median
    #
    summary(lm(op.blq$obs ~ against.blq))
  } # op.blq <- remove BLQ observations
  against <- op$post_median # "x" or predictor variable
  {
    plot.lim = c(0,max(max(op$obs),max(against)))
    plot(against, op$obs, type = "p", ylim = plot.lim, xlim = plot.lim,
         , xlab = "posterior concentration; regression w/out BLQ", ylab = "observed concentration"
         , main = paste("posteror median\n", run_name))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
  } # ... and plot obs vs. against
  {
    {
      inter = lm(op.blq$obs ~ against.blq)[1]$coefficients[1]
      sl = lm(op.blq$obs ~ against.blq)[1]$coefficients[2]
      lines(plot.lim,
            inter + sl*plot.lim
            , lty = 7, lwd = 3, col = "grey"
      )
    } # plot reg line w/out BLQ
    # abline(h = c(BLQ), col = "grey")
  } # ... overlay regression line w/BLQ removed!!!
  {
    hist(op$obs - against, breaks = 73)
    op$id[which(op$obs - against < -7)]
    # [1] 12 37 49  5 50 54  6  7
    idcheck <- op[which(op$id == 7),]; lines(idcheck$post_median, idcheck$obs, col = "red", type = "p")
    
  } # 3 look for the outliers
  # ------------------------------------------------------- POPULATION FITS ---
  against <- op.blq$pop_median # (op %>% filter(obs > 4))$pop_median
  obs <- op.blq$obs # (op%>% filter(obs > 4))$obs 
  # against <- log10(against)
  # obs <- log10(obs)
  {
    plot.lim = c(0,max(max(obs),max(against)))
    plot(against, obs, type = "p", ylim = plot.lim, xlim = plot.lim
         , xlab = "population concentration; if BLQ, obs vs. post median", ylab = "observed concentration"
         , main = paste("pop_median\n",run_name)) # , pch = as.character(op$id))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
  } # obs vs. pred
  # ---
  {
    op.blq <- op %>%
      filter(obs == BLQ) %>% # find BLQ measurements
      # sigma is from main.rs
      mutate(obs = if_else(post_median < BLQ + 1.93*(0.1 + 0.15*obs),
                           post_median, obs))
    lines(op.blq$pop_median,
          op.blq$obs, col = "red", type = "p") 
    abline(h = c(BLQ), col = "grey")
  } # for (obs ==  BLQ) if CI95%(BLQ) covers the posterior pred, then use post instead of obs to visualize data
  # ---
  summary(lm(obs ~ against))
  {
    inter = lm(obs ~ against)[1]$coefficients[1]
    sl = lm(obs ~ against)[1]$coefficients[2]
    lines(plot.lim,
          inter + sl*plot.lim
          , lty = 7, lwd = 3, col = "grey"
    )
    # abline(h = c(BLQ), col = "grey")
  } # summary obs~against
  
} # op plot -- median
{
  op <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde_arc/op.csv")
  run_name <-  "vpicu SDE arc; ID 5 #527974024"; # ; subjects 39 and 64"  # "vpicu ODE; nCRCL"
  #op <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde_sigma_only/op.csv")
  #run_name <- "vpicu SDE sigma only"
  #
  BLQ = 4.0;
  {
    op.blq <- op %>% filter(obs > BLQ)
    against.blq <- op.blq$post_mean
  } # op.blq <- remove BLQ observations
  against <- op$post_mean
  {
    plot.lim = c(0,max(max(op$obs),max(against)))
    plot(against, op$obs, type = "p", ylim = plot.lim, xlim = plot.lim,
         , xlab = "posterior expected concentration", ylab = "observed concentration"
         , main = paste("post_mean\n", run_name
         ))
    lines(c(0,max(plot.lim)),c(0,max(plot.lim)),lty = 3,lwd = 2, col = "grey")
    # points(against, op$obs, col = "red", pch = 19);
  } # ... and plot obs vs. against
  summary(lm(op.blq$obs ~ against.blq))
  {
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
  plot(op$time,(op$post_mean - op$obs)); abline(v = c(101.8, 227.8), h =c(0))
  abline(h=c(9.399999, 19.5, 8.3, 11.2, 15.9)) # id5 first two observations
  abline(h = c(14.89999999, 15.0, 13.8, 12.4), col = "red") # id5 last two observations 
  # find outliers
  {
    hist(op$obs - against, breaks = 73)
    id_outs <- op$id[which(abs(op$obs - against) > 3)] %>%
      unique() %>% sort()
    id_in <- unique(op$id)[!(unique(op$id) %in% id_outs)]
    # [1] 12 37 49  5 50 54  6  7
    # [1] 5  5  5 39  50 50 > 10
    idcheck <- op[which(op$id == 39),]; lines(idcheck$pop_mean, idcheck$obs, col = "cyan", type = "p")
    #
    id_outs <- op$id[which(op$obs > 25 & op$pop_mean < 20)]
    # [1] 18 37 37 37 39  5 50
    lines(c(20,20),c(25,30))
    lines(c(15,20),c(25,25))
    #
    op.in <- op %>% filter(id %in% id_in)
    points(op.in$pop_mean, op.in$obs) # , col = "green")
    op.out <- op %>% filter(id %in% id_outs)
    op.5 <- op %>% filter(id == 5)
    points(op.out$pop_mean, op.out$obs); #, col = "cyan")
    op.ttt <- op %>% filter(time < 48)
    points(op.ttt$pop_mean, op.ttt$obs, col = "cyan")
  } # 3 look for the outliers
  # ------------------------------------------------------- POPULATION FITS ---
  op.blq <- op %>%
    mutate(obs = ifelse(obs == 4.0, post_mean, obs))
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
      mutate(obs = if_else(post_mean < BLQ + 1.93*(0.1 + 0.15*obs),
                           post_mean, obs))
    lines(op.blq$pop_mean,
          op.blq$obs, col = "red", type = "p") 
    abline(h = c(BLQ), col = "grey")
  } # for (obs ==  BLQ) if CI95%(BLQ) covers the posterior pred, then overplotplot post (instead of obs, to visualize data)
  # ---
  summary(lm(obs ~ against))
  {
    inter = lm(obs ~ against)[1]$coefficients[1]
    sl = lm(obs ~ against)[1]$coefficients[2]
    lines(plot.lim,
          inter + sl*plot.lim
          , lty = 7, lwd = 3, col = "grey"
    )
    # abline(h = c(BLQ), col = "grey")
  } # summary obs~against
  #
  points(op$pop_mean[which(op$id == 2)], # unique(id_outs)[2])],
        op$obs[which(op$id == 2)], pch = 19, col = "wheat2") # unique(id_outs)[2])])
  points(op$pop_mean[which(op$id == 3)], # unique(id_outs)[2])],
         op$obs[which(op$id == 3)], pch = 19, col = "steelblue1") # unique(id_outs)[2])])
  lines(op$post_mean[which(op$id == 5)], # unique(id_outs)[2])],
         op$obs[which(op$id == 5)], pch = 19, col = "red") # unique(id_outs)[2])])
  points(op$pop_mean[which(op$id == 6)], # unique(id_outs)[2])],
         op$obs[which(op$id == 6)], pch = 19, col = "blue") # unique(id_outs)[2])])
  points(op$pop_mean[which(op$id == 7)], # unique(id_outs)[2])],
         op$obs[which(op$id == 7)], pch = 19, col = "brown") # unique(id_outs)[2])])
  points(op$pop_mean[which(op$id == 8)],
         op$obs[which(op$id == 8)], col = "green", pch = 19)
  
  points(op$pop_mean[which(op$id == unique(id_outs)[3])],
         op$obs[which(op$id == unique(id_outs)[3])], pch = 19, col = "cyan")
  points(op$pop_mean[which(op$id == unique(id_outs)[5])],
         op$obs[which(op$id == unique(id_outs)[5])], col = "red")
  points(op$pop_mean[which(op$id == unique(id_outs)[6])],
         op$obs[which(op$id == unique(id_outs)[6])], pch = 19, col = "blue")
  points(op$pop_mean[which(op$id == unique(id_outs)[7])],
         op$obs[which(op$id == unique(id_outs)[7])], pch=19, col = "wheat2")
  } # op plot -- mean
{
  # run wtCov.R
  theta_0 <- theta_ode;
  theta_ode <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode_arc/theta.csv")
  DDD <- matrix(c(theta_0,theta_ode),byrow=T,ncol=length(theta_0)); colnames(DDD) <- names(theta_ode); DDD
  
  theta_sde_0 <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde/theta_w_sigma.csv")
  theta_sde <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde/theta.csv")
  theta_sde_s <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde_sigma_only/theta.csv")
  theta <- theta_ode; # theta_sde_s
    summary(theta)
    np.stats(theta)
    cov2cor(np.stats(theta)[[5]])
    
} # inspect theta

{
  theta_1to6 <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/prior_first_six.csv")
  summary(theta_1to6)
}


{
  theta_sde <- read_csv("/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode_arc/theta.csv") %>%
    add_column(ske = 0.25, svol = 0.0000025) %>%
    relocate(prob, .after = svol)
  write_csv(theta_sde,"/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_ode_arc/theta_add_sigma.csv", na = "." )
  write_csv(theta_sde,"/Users/wyamada/src/lapk/PMcore/examples/vpicu_for_grant_prop/output_sde/theta_w_sigma.csv", na = "." )
  #
  # to run the sde:
  # 1) set proper equation
  # 2) add the extra parameters
  # 3) !!! set the output path !!!
  # 4) *** set the adaptive gtid to ONLY find new points in the dimensions of sigmas ***
  
} # write theta_w_sigma.csv, prior for sde run

setwd(goback)

{
  # rust
  # let vanc_conc = x[2]/vol;
  # let k_e_mean = ke0 * ke_vs_crcl * crcl * (wt/70.0).powf(-0.25)
  #     * (1.0 + alpha_ke/(1.0 + ((conc_central_eff - vanc_conc)/ke_slope).exp()));
  # dx[0] =  (k_e_mean - x[0])/tau_kel_reversion; // 168  336  504  672  840 1008 1176 1344 1512 1680
  # let k_e = x[0];
  crcl = 100.0; wt = 7.4
  vanc_conc <- seq(0,35)
  ke0 = 1.0
  alpha_ke = 2.0
  conc_central_eff = 20
  ke_slope = 1.0
  ke_vs_crcl = 1.0
  a_obs = ke0 * ke_vs_crcl * crcl * ( wt / 70.0 )^(-0.25)
  kel_mu = a_obs *
    (1.0 + alpha_ke/(1.0 + exp((conc_central_eff - vanc_conc)/ke_slope)))
  plot(vanc_conc,kel_mu, ylim = c(a_obs*(1.0 - alpha_ke), a_obs*(1.0 + alpha_ke)))
  alpha_ke = -1
  kel_mu = a_obs *
    (1.0 + alpha_ke/(1.0 + exp((conc_central_eff - vanc_conc)/ke_slope)))
  lines(vanc_conc,kel_mu, ylim = c(a_obs*(1.0 - alpha_ke), a_obs*(1.0 + alpha_ke)))
  
  
  tau_kel_reversion = 2.4; # Hrs
  
} # -- k_el equation in Rust and R

{
  ke0 =1.0; alpha_ke = 2.0; conc_arc = 10; vanc_conc = seq(1,40)
  plot(vanc_conc, ke0 * (1.0 + alpha_ke/(1.0 + exp((conc_arc - vanc_conc))) ) )
} # -- Playing w/functions for drift equation


# ------ #
{
  # TDs ANOVA example
  data <- data.frame(race = c(rep(c("Black"),8), rep(c("White","Hispanic","Asian"),each=7)),
                     eat_out = c(14,0,14,13,7,12,4,1,11,6,2,4,9,12,14,15,11,3,13,15,7,5,10,8,13,0,15,12,15));
  summary(data);
  data %>% group_by(race) %>%
    summarise(mean = mean(eat_out), sd = sd(eat_out))
  boxplot(eat_out ~ race, data = data, "Cultural proclivity to eat out", xlab = "Ethnicity", ylab = "Times per wk")
  aov(eat_out ~ race, data = data) # print out basic results
  mmm <- aov(eat_out ~ race, data = data) # save aov class object for further analysis
  summary(mmm)
} # TD 1-way ANOVA example
{
  t.test(x <- rnorm(n = 23, mean = 1, sd = 1.0), y = NULL,
         alternative = "less", # c("two.sided", "less", "greater"),
         mu = 0, paired = FALSE, var.equal = FALSE,
         conf.level = 0.95) # , ...)
} # TD t.test sample
{
  # You can run your R codes here.
  link="https://docs.google.com/spreadsheets/d/e/2PACX-1vR3Joryy90fkhlL7ab2I_BUhu5BmK6UTyi44qRxDXm2Inml4ZQN1zGF1suDsgFIWSdERQ9YAAIAGfJH/pub?output=csv"
  MPdata=read.csv(link, header=TRUE)
  #This is the sample of a data set from Row 10 to Row 21
  data=MPdata$Mpt.of.Hor
  threshold <- quantile(data,0.75) - IQR(data)
  library(tidyverse)
  # ddd <- as.data.frame(matrix(c(rnorm(n = 23, mean = 0, sd = 1.0),-7,7), byrow = T, ncol = 1))
  ddd <- as.data.frame(data);
  names(ddd) <- c("my_data")
  hist(ddd$my_data,breaks = 17)
  ddd <- ddd %>% filter(my_data > -threshold) %>% filter(my_data < threshold)
  hist(ddd$my_data,breaks = 17)
} # TD get rid of outliers
{
  x <- seq(1,100)
  x %in% c(2,3,5,7,11,13)
  x[x %in% c(2,3,5,7,11,13)]
  which(x %in% c(2,3,5,7,11,13,17))
} # TD example using %in%
{
  library(factoextra)
  library(cluster)
  
  link="https://docs.google.com/spreadsheets/d/e/2PACX-1vR7xrN8VKL8LUZJPYkKn303V8Tv3js6SmTn5KCiK0UO7uWEh66uHXBkczgPgo-tbO3s8iiU9GKbIxgE/pub?output=csv"
  data=read.csv(link, header=TRUE)
  #This is the sample of a data set from Row 40 to Row 63
  x=data $ Hand
  y=data $ Height
  #
  plot(x,y); abline(v = c(5,12))
  new_data <- data %>% filter(Hand > 5) %>% filter( Hand < 12)
  points(new_data$Hand,new_data$Height, col = "cyan", pch = 17)
  
  set.seed(1)
  km <- kmeans(data %>% select(Height,Hand), centers = 4)
  fviz_cluster(km, data = df[, -5],
               palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
               geom = "point",
               ellipse.type = "convex", 
               ggtheme = theme_bw()
  )
} # TD remove outliers in 2-D: She needs a clustering algorithm. But my R is not up to date w/online packages.


