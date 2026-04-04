use std::f64::consts::E;

use anyhow::Result;
use logger::setup_log;
use pharmsol::builder::SubjectBuilder;
use pmcore::prelude::*;
use settings::{Parameters, Settings};
// use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};

const MIC:f64 = 10.0;
const LLQ:f64 = 4.0;

fn main() -> Result<()> {
    let _eq = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            // automatically defined
            fetch_params!(p, ke0, kcp0, kpc0, v0, alpha_ke, conc_peri_eff,two2one); // , conc_peri_eff); // , alpha_ke, conc_peri_eff);
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t
            /* 
               kcp is increased under conditions of inflammation, e.g. due to blood brain barrier weakening -- indicated by elevated C-protein
               there is no access to C-protein levels as a covariate ... maybe we can use kcp ~ 1/(time above MIC) 
               ... this would convert the model from two to 1 compartment over time of treatment.
               kcp : kcp_0 -> kcp_final; kcp_0 > kcp_final, we can assume a linear approach w.r.t time>MIC
               r.v.s kcp_0, kcp_final, rate_of_decay, MIC
               kcp = kcp_final + (kcp_0 - kco_final)/(time > MIC)
            
               kel is increased (often) for critically ill patients, augmented renal clearance (ARC; 30-65% of ICU patients)
               in our dataset, mean crcl is about 135 ... a typcal threshold for ARC is 130 ml/min/1.73m^@
            
              note: typically want to have troughs that are 10-20 mg/L
            */
            let vol = v0 * (wt/70.0);
            let k_pc = kpc0;
            // let mut k_e = x[4]; // alpha_ke * ke0; // k_e will be wt and CRCL normalized after ARC adjustment
            // let mut k_e_mean = ke0;
            
            let vanc_conc = x[0]/vol;
            let mut _arc = true;
            let mut _check_arc = true;

            // adjust k_e and k_cp as necessary after 6 min
            // if t > 0.1 {
              // kel adjustmants for ARC

              // if vanc_conc > conc_peri_eff { // if estimated concentration fell below MIC_0 the dose will be increased
                // arc = true;
                // let conc_peri_eff = 18.0; // 12.69;
                let k_e_mean = ke0 * (1.0 + alpha_ke/(1.0 + ((conc_peri_eff - vanc_conc)/2.0).exp())) ; //alpha_ke * ke0;
                // 0.53 * (1.0 + alpha_ke/(1.0 + ((conc_peri_eff - vanc_conc)/2.0).exp())) ; //alpha_ke * ke0;
              // } else {
              //     k_e_mean = ke0;
              // }
              
              /*
              if rateiv[0] > 0.0 {
                if check_arc == true {
                  check_arc = false; // only check at start of a dose
                  if vanc_conc > MIC {
                    k_e = ke0;
                  }
                }
              } else {
                check_arc = true;
              }
              if arc == true {
                k_e = alpha_ke * ke0;
              }  
              */
            // } // if t > 0.1
            
            // k_e = k_e * (wt/70.0).powf(-0.25) * (crcl/120.9); // 
            let k_e = x[4] * (wt/70.0).powf(-0.25) * 0.64 * crcl / 120.9;
            let k_cp = kcp0 / (1.0 + x[2]/two2one); // 336.0); // make sure at T=0 k_cp doesn't go to infty

            // user defined two-comp model
            let d_mg = rateiv[0] - (k_e + k_cp) * x[0] + k_pc * x[1];
            dx[0] =  d_mg;
            dx[1] = k_cp * x[0] - k_pc * x[1];
            
            // time>MIC, running AUC, other stats
            if vanc_conc > MIC {
                dx[2] = 1.0;
            } else {
                dx[2] = 0.0;
            }
            dx[3] = d_mg/vol; // AUC total
            dx[4] =  (x[4] - k_e_mean)/2400.0; // 168  336  504  672  840 1008 1176 1344 1512 1680         
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _kcp0, _kpc0, _v0, _alpha_ke); // , alpha_ke, _conc_peri_eff);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            x[3] = 0.0;
            x[4] = ke0; // 0.53;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ke0, _kcp, _kpc, v0, _alpha_ke); // , _alpha_ke, _conc_peri_eff);
            fetch_cov!(cov,t,wt); // , _crcl); // automatically interpolates, so you need t

            // let k_e = x[4] * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = v0 * (wt/70.0);

            y[0] = x[0]/vol;
        },
        (5, 1),
    );

    let eq = equation::SDE::new(
        |x, p, t, dx, rateiv, cov| {
            /*
            prior to working on drift:
            fetch_params!(p, ke0, kcp, kpc, v0, _ske, _svol);
            fetch_cov!(cov,t,wt,crcl);
            dx[0] = ke0 - x[0]; // mean reverting sde
            dx[1] = v0 - x[1];
            let ke = x[0]; // use ke = ke0, if SDE in only on volume.
            let _vol = x[1]* (wt/70.0);
            // let kpc = well * kcp;
            // let norm_wt = wt/70.0;
            // let kel = ke * norm_wt.powf(-0.25) * (0.2145/scr).powf(1.1776);
            let k_e = ke * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            dx[2] = rateiv[0] - ( k_e + kcp) * x[2] + kpc * x[3];
            dx[3] = kcp * x[2] - kpc * x[3];
            */
            fetch_params!(p, ke0, kcp0, v0, alpha_ke, conc_central_eff, tau_kel_reversion, ke_slope, ke_vs_crcl, conc_peri_eff, tau_auc, tau_mic, tau_p_periph_eff, _p_periph_eff_0, _ske, _svol); // , conc_peri_eff); // , alpha_ke, conc_peri_eff);
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t

            dx[1] = v0 - x[1]; // mean reverting sde
            let vol = x[1] * (wt/70.0);
            let vanc_conc = x[2]/vol;
            let k_e_mean = ke0 * ke_vs_crcl * crcl * (wt/70.0).powf(-0.25)
                * (1.0 + alpha_ke/(1.0 + ((conc_central_eff - vanc_conc)/ke_slope).exp()));
            // dx[0] = ke0 - x[0]; // mean reverting to ke0
            dx[0] =  (k_e_mean - x[0])/tau_kel_reversion; // 168  336  504  672  840 1008 1176 1344 1512 1680
            let k_e = x[0]; //  * (wt/70.0).powf(-0.25) * ke_vs_crcl * crcl; // k_e_mean is normalized to wt and crcl
            /*
                For k_e we can have two more r.v.s, slope of effect in k_e_mean and integration tau in mean reversion
            */
            // /*
                let a_kcp = x[7] / (1.0 + x[5]); // or this: 
                // let a_kcp = x[7] * conc_peri_eff / (conc_peri_eff + x[5]);
                let k_cp = kcp0 * a_kcp;
                let k_pc = 1.0; 
            // */ // This block is for the effect on kcp; below code rewrites the above.
            /* // parameter names are still w.r.t effect on k_cp, so the only change outside of this function is
            // the parameter ranges.
            let a_k_cp = x[7];
            let a_kpc = x[5]/conc_peri_eff; // ~ (AUC_Dt /IC)
            */


            // user defined two-comp model
            let d_mg = rateiv[0] - (k_e + k_cp) * x[2] + k_pc * x[3];
            dx[2] =  d_mg;
            dx[3] = k_cp * x[2] - k_pc * x[3];

            // time>MIC, running AUC, other stats
            dx[5] = (d_mg/vol) - x[5]/tau_auc; //  "/4.8;" // AUC(t-24) total  
            let tau_gt_conc_eff = tau_mic; // 2.4; // tau_auc; // 2.4; 2.4 was for all subjects prior to 8
            if vanc_conc >= conc_peri_eff {
                dx[4] = 1.0 - x[4]/tau_gt_conc_eff; // 33.6Hr is a leaky integrator w/5*tau ~ 1 week
                dx[6] = -1.0 * x[6]/tau_gt_conc_eff; 
            } else {
                dx[4] = -1.0 * x[4]/tau_gt_conc_eff;
                dx[6] = 1.0 - x[6]/tau_gt_conc_eff;
            }         
            // if t > 24.0 { // this has to start from t=0, so initialize x_6 = 1.0e-8 float at t=0.
                if x[7] > 0.05 {
                    dx[7] = x[6]/(x[4] + x[6]) - x[7] / tau_p_periph_eff;
                } else {
                    dx[7] = - x[7] / tau_p_periph_eff; // if EFF resolved, do not allow to recur
                }
            // } else {
            //    dx[7] = 0.0;
            // }     
        },
        |p, d| {
            fetch_params!(p, _ke0, _kcp, _v0, _alpha_ke, _conc_central_eff, _tau_kel_reversion, _ke_slope, _ke_vs_crcl, _conc_peri_eff, _tau_auc, _tau_mic, _tau_p_periph_eff, _p_periph_eff_0, ske, svol);
            d[0] = ske; // 0.0; // ske * ke0;
            d[1] = svol; // 0.0; // svol * v0;
            // the above increments MUST match the state increments of x
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, t, cov, x| {
            fetch_params!(p, ke0, _kcp0, v0, alpha_ke, conc_central_eff, _tau_kel_reversion, ke_slope, ke_vs_crcl, _conc_peri_eff, _tau_auc, _tau_mic, _tau_p_periph_eff, p_periph_eff_0, _ske, svol); // , conc_peri_eff); // , alpha_ke, conc_peri_eff);
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t
            /*
            let normal_ke = Normal::new(ke0, ske*ke0).unwrap();
            x[0] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            */
            let normal_v = Normal::new(v0, svol*v0).unwrap();
            x[1] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            
            x[0] = ke0 * (1.0 + alpha_ke/(1.0 + (conc_central_eff/ke_slope).exp())) * ke_vs_crcl * crcl * (wt/70.0).powf(-0.25);
            // x[1] = v0;
            x[2] = 0.0; // central compartment
            x[3] = 0.0; // peripheral compartment
            x[4] = 0.0; // time >= conc_peri_eff
            x[5] = 0.0; // (total AUC / 4.8) -> AUC(t-24Hr)
            x[6] = 1.0e-8; // time < conc_peri_eff
            x[7] = p_periph_eff_0;
        },
        |x, _p, t, cov, y| {
            // fetch_params!(p, _ke0, _kcp, _kpc, v0);
            fetch_cov!(cov,t,wt); // , crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = x[1] * (wt/70.0);

            y[0] = x[2]/vol;
        },
        (8, 1),
        271,
    );
/*
    let supp_point = vec![0.0676, 0.00108, 2.06, 55.1, 4.50, 18.4, 229.0, 0.396]; 
    
    let subject = Subject::builder("999")
        .infusion(0., 150.0, 0, 1.0)
        .observation(0.0, -99.0,0)
        .repeat(420, 0.5)
        .build();

    let sim = eq.estimate_predictions(&subject, &supp_point); // simulator
    // sim is an array of predictions: time, obs

   for s in sim{
    println!("time {} : {}", s.time(), s.prediction());
   }
*/
    let mut settings = Settings::new();

/*
    let params = Parameters::builder()
        .add("ke0", 1.023e-6, 1.027, true)
        .add("kcp", 1.0e-8, 1.5, true)
        // .add("kpc", 0.01, 4.0, true) // fix at 1.0 because all that matters is ratio
        .add("v0", 14.7, 75.1, true)
        .add("alpha_ke", -0.5, 4.5, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("conc_central_eff", MIC, 3.0*MIC, true)
        .add("tau_kel_reversion", 4.8,67.2 , true)
        .add("ke_slope", 1.5, 6.0, true)
        .add("conc_peri_eff", MIC, 3.0*MIC, true)
        .add("tau_auc", 4.8, 33.6, true)
        .add("tau_p_periph_eff", 4.8, 33.6, true)
        .add("p_periph_eff_0", 1.0e-12, 1.0, true)
        // .add("ncrcl", 20.0, 150.0, true)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 0.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();
 */    // ID 2 and 3

/* ID 5 temp values for first 6 observations
0.16842868441205025,0.6186445989606953,59.12515146613121,2.0393963661193846,15.019012093544006,
6.8912396383285515,3.087835167527199,0.0022051507949068425,23.218577980995178,14.402883982658388,
10.4601893863678,5.298442543029785,0.9373295783996582,1

ID 5 middle observations
0.02086441670963287,1.1940279259089468,59.12837776947021,1.396260947098732,18.37471956611872,
15.88111696639061,2.4628564720749857,0.010865467841494083,15.112705895018578,16.283026878643035,
24.604715196418763,27.4619341135025,0.6004384756088257,1

0.020864515166282654,1.1934531061649323,59.11203517913818,1.4309060943126677,25.968996047973633
15.825645852088929,2.4720661640167236,0.010862913467884063,15.258760118484497,16.28193719983101,
24.622047674655914,27.426012539863585,0.2510455846786499,1

 */
/* best description for ID 5

    let params = Parameters::builder()
        .add("ke0", 5.92768892e-4, 5.92768893e-4, true)
        .add("kcp0", 2.0181786680221, 2.0181786680222, true)
        .add("v0", 119.60780, 119.60781, true)
        .add("alpha_ke", 1.69322331571, 1.69322331572, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("conc_central_eff", 11.565801143, 11.565801144, true)
        .add("tau_kel_reversion", 1.2501105308,1.2501105309, true)
        .add("ke_slope", 0.5532256007194, 0.5532256007195, true)
        .add("ke_vs_crcl", 0.30805915, 0.30805916, true)
        .add("conc_peri_eff", 7.75372576713,7.75372576714, true)
        .add("tau_auc", 1.667665243148803, 1.667665243148804, true) // 10.4 to 2.4
        .add("tau_mic", 2.644729700088, 2.644729700089, true)
        .add("tau_p_periph_eff", 3.222090, 3.222100, true)
        .add("p_periph_eff_0", 0.86, 0.8628, true)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 0.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();

ke0,kcp0,
0.00058462845993042,1.9300028228759765,
v0,
119.60780568122864,
alpha_ke,conc_central_eff,tau_kel_reversion,ke_slope,ke_vs_crcl,
1.6932233157157899,11.56580114364624,1.2501105308532707,0.5532256007194519,0.3080591559410095,
conc_peri_eff,tau_auc,tau_mic,tau_p_periph_eff,p_periph_eff_0,prob
7.75372576713562,1.6676652431488035,2.6447297000885013,3.354799690246582,0.8614429724216461,1
 */

// FITS ID 2 (op plot w/slope approx. 1, and small shift to underprediction; but
//.   still w/+ slope in (pred - out) ~ time, approx. (+1 - -1)/171Hrs)
// ke0,kcp0,v0,
// alpha_ke,conc_central_eff,tau_kel_reversion,ke_slope,ke_vs_crcl,
// conc_peri_eff,tau_auc,tau_mic,tau_p_periph_eff,p_periph_eff_0,prob
//
// 0.9853662637182236,0.008041634697699546,38.87368631362915,
// 0.13532614707946777,24.79432713985443,1.3187536239624023,0.9581841584613323,0.0014065953274965284,
// 23.284181237220764,43.27419118881226,18.824203491210938,58.64706716537476,0.5600688934326172,1
//
// FITS ID 3 (only two obs, not perfect, not perfect, but w/approx -0.145 pred-obs error)
// 0.19902523983359335,0.014635046225619315,31.540117263793945,
// 0.10625600814819336,14.706377744674683,8.262526702880859,0.15960874462008479,0.006551186282753944,
// 14.514846563339233,63.44421129226685,50.554429721832285,4.794456052780152,0.17287707328796387,1
//
// FITS ID 5 (slope near 1), error in (-4,4)
    /*
     let params = Parameters::builder()
        .add("ke0", 1.82180e-3, 1.8218034e-3, true)
        .add("kcp0", 1.0e-2, 1.0, true)
        // .add("kpc", 0.01, 4.0, true) // fix at 1.0 because all that matters is ratio
        .add("v0", 100.0, 200.0, true)
        .add("alpha_ke", 0.0, 2.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("conc_central_eff", LLQ, MIC, true)
        .add("tau_kel_reversion", 1.2, 12.0, true) // 1.2,(4.8, 16.0)
        .add("ke_slope", 1.0, 4.0, true) // 2.0
        .add("ke_vs_crcl", 1.0e-2, 1.0, true)
        .add("conc_peri_eff", MIC, 2.5*MIC, true) // 2.5*MIC
        .add("tau_auc", 1.2, 48.0, true) // 9.6
        .add("tau_mic", 1.2, 48.0, true) // 4.8
        .add("tau_p_periph_eff", 1.2, 72.0, true) // 4.8
        .add("p_periph_eff_0", 0.0, 1.0, true)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 0.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();
    */
// range above gets this, nice fit w/slope almost 1 and err in (-4,5):
// 0.0018218021402074526,0.5832987687587737,154.68374252319336,
// 0.3833851337432862,9.18260145187378,9.559475784301759,1.159221792221069,0.1383254086971283,
// 10.671962022781372,19.351201400756835,9.106321506500244,28.16787202835083,0.2264639377593994,1
//
// restricted to above point and got this w/(p(periph eff) in (0,1))
// 0.001821800010631876,0.5832987687167709,154.68374280158494,
// 0.3833887740838528,9.182601460600615,9.55947883309126,1.1592217919900196,0.13832540903716903,
// 10.671962021821093,19.351201401305868,9.10632150602916,28.167872028336106,0.09113597869873047,1

// /* Final parameter ranges for ID 5: 3/31/2026 ... add the stochastic element:
     let params = Parameters::builder()
        .add("ke0", 1.82180e-3, 1.82180428e-3, true)
        .add("kcp0", 5.832987687e-1, 5.832987688e-1, true)
        // .add("kpc", 0.01, 4.0, true) // fix at 1.0 because all that matters is ratio
        .add("v0", 154.683742, 154.683743, true)
        .add("alpha_ke", 0.38338, 0.38339, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("conc_central_eff", 9.1826014, 9.1826015, true)
        .add("tau_kel_reversion", 9.55947, 9.55948, true) // 1.2,(4.8, 16.0)
        .add("ke_slope", 1.15922179, 1.15922179444, true) // 2.0
        .add("ke_vs_crcl", 1.38325408e-1, 1.38325409394e-1, true)
        .add("conc_peri_eff", 10.67196202, 10.67196202556, true) // 2.5*MIC
        .add("tau_auc", 19.35120140, 19.3512014015, true) // 9.6
        .add("tau_mic", 9.106321506, 9.106321507, true) // 4.8
        .add("tau_p_periph_eff", 28.1678720283, 28.1678720284, true) // 4.8
        .add("p_periph_eff_0", 0.0, 1.0, true)
        .add("ske", 0.1, 2.0, true)
        .add("svol", 0.0001, 0.00025, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();
// */

/* First, naive optimization to guess good set point/s:
    // tau imply equilibrium w/in 6Hrs to 2 weeks
    // alpha in (-1, 2.0) ~ (No renal function,  3x normal)
    //.   renal function goes from kel0 to kel0*(1.0 + alpha)
    // p_peripheral_eff ~ initial relative magnitude of k_cp [unobserved covariate]

    let params = Parameters::builder()
        .add("ke0", 1.0e-6, 2.0, true)
        .add("kcp0", 1.0e-8, 1.0, true)
        // .add("kpc", 0.01, 4.0, true) // fix at 1.0 because all that matters is ratio
        .add("v0", 20.0, 190.0, true)
        .add("alpha_ke", -1.0, 10.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("conc_central_eff", LLQ, 2.5*MIC, true)
        .add("tau_kel_reversion", 1.2, 66.4, true) // 1.2,(4.8, 16.0)
        .add("ke_slope", 1.0e-2, 4.0, true) // 2.0
        .add("ke_vs_crcl", 1.0e-3, 2.0, true)
        .add("conc_peri_eff", MIC, 3.5*MIC, true) // 2.5*MIC
        .add("tau_auc", 1.2, 48.0, true) // 9.6
        .add("tau_mic", 1.2, 48.0, true) // 4.8
        .add("tau_p_periph_eff", 9.6, 72.0, true) // 4.8
        .add("p_periph_eff_0", 0.0, 1.0, true)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 0.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();
*/
    settings.set_parameters(params);

    settings.set_prior_sampler("sobol".to_string());
    settings.set_prior_points(147000);
    settings.set_prior_seed(347);

    settings.set_cycles(1000);
    // settings.set_error_poly((0.1, 0.075, -0.00165, 0.0)); // MN uses 0.1,0.15 ... CV%<0.1 is an acceptiable assay ... so 0, 0.1, ... is probably "right" to use for comparing SDE to ODE solutions
    settings.set_error_poly((0.0, 0.15, 0.0, 0.0));
    settings.set_error_type(ErrorType::Add);
    settings.set_error_value(2.0*LLQ);
    settings.set_idelta(1.0);

    // for ODE use this block:
    /*
        settings.set_output_path("examples/vpicu_for_grant_prop/output_ode_arc"); // THIS LINE OVERWRITES THIS DIRECTORY !!!
        settings.set_prior_sampler("sobol".to_string());
        settings.set_prior_points(46657);
        settings.set_prior_seed(347);
        // settings.set_prior(settings::Prior {
        //    sampler: "sobol".to_string(),
        //    points: 16384,
        //    seed: 347,
        //    file: None, // Some(String::from("examples/vpicu_for_grant_prop/output_ode/theta.csv")),
        // });
    */

    // for SDE use this block (Verify AG is edited to expand only in dimensions of sigma: ___ YES ___):
    // /*
        settings.set_output_path("examples/vpicu_for_grant_prop/output_sde_arc"); // THIS LINE OVERWRITES THIS DIRECTORY !!!
        settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_ode_arc/theta_add_sigma.csv")));
        // settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/prior_first_six.csv")));
        // settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_ode/theta_w_sigma.csv")));
        // settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_ode_arc/theta.csv")));
        //
        // to optimize ONLY the sigmas, edit src/routines/expansion/adaptive_grid.rs to only expand in the dimentions of sigma
        //
    // */

    setup_log(&settings)?;
    let data = data::read_pmetrics("examples/vpicu_for_grant_prop/vpicu_5.csv")?; // subj1to6.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit().unwrap();
    result.write_outputs()?;

    Ok(())
}
