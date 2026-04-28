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

/* fn drug_effect_on_k() returns the k + a standard e50 model
 */
fn drug_effect_on_k(alpha:f64, p:f64, e50:f64, e_now:f64, slope:f64) -> f64 {
    // p in (0,1)
    // other parameter range and error checking is not necesssary
    if p < 0.0 || p > 1.0 {
        println!("invalid p <p>; returns 0.0");
    }
    1.0 + alpha * p /(1.0 + ((e50 - e_now)/slope).exp())
}

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
            fetch_params!(p, v0, ke0, kcp0, kpc0 // non-infected state
                , ke_vs_crcl // dependence on covariates
                , _p_periph_eff_0 // prob of infection effect on pk parameters at start of treatment
                , tau_p_periph_eff // x_7(t=0); relative strength of infection
                , _p_cent_eff_0
                , tau_p_cent_eff
                , tau_mic // strength of infection (t) is ~ percent time above or below MIC
                , conc_central_eff // drug E50 for k_e and k_cp
                , alpha_ke , ke_slope 
                , alpha_kcp, kcp_slope
                , tau_auc // peripheral drug exposure is related to integrated difference in drug concentration between periphery and center
                , conc_peri_eff // drug E50 for kpc
                , alpha_kpc // kpc_slope=1.75
                , kpc_e50
                , tau_kel_reversion, tau_kcp_reversion, tau_kpc_reversion
                // , _ske
                // , _svol
                );
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t

            dx[1] = v0 - x[1]; // mean reverting sde
            let vol = x[1] * (wt/70.0);
            let vanc_conc = x[2]/vol;

            let p_inf_periph = 1.0 /(1.0 + ((x[7] - 0.5)/0.08).exp()); // map x[7] in (-infty,+infty) to (1,0); for effect fully on until treatment is efficacious; 
            let p_inf_central = 1.0 /(1.0 + ((0.5 - x[11])/0.08).exp()); // map x[11] to (0,1)

            let k_e_mean = ke0 * ke_vs_crcl * crcl * (wt/70.0).powf(-0.25) // k_e_mean is normalized to wt and crcl
                * drug_effect_on_k(alpha_ke,p_inf_central,conc_central_eff,vanc_conc,ke_slope);
            dx[0] =  (k_e_mean - x[0])/tau_kel_reversion;
            let k_e = x[0];

            /*
                // let a_kcp = x[7] * conc_peri_eff / (conc_peri_eff + x[5]);
                let a_kcp = x[7] / (1.0 + x[5]); // or this: 
                let k_cp = kcp0 * a_kcp;
                let k_pc = 1.0; 
            */ // This block is for the effect on kcp only; below code rewrites the above to have effect on kcp and kpc.
            /* This block has notes
            4/3/2026 to 4/10/2026
                k_cp should be modeled similar to k_el, with a delayed response to a concentration dependent expected effect (exposure)
                k_pc should be modeled with an AUC/MIC model, b/c the amount that enters the periphery (exposure) is
                    dependent on the integrated difference of peripheral and blood concentrations (but peripheral
                    concentration is an `imagined' variable.)
                note: https://www.sciencedirect.com/topics/medicine-and-dentistry/vancomycin 
                      https://pharmacologymentor.com/pharmacology-of-vancomycin/
                      1) vancomycin is large, not readily absobed. AUC/MIC effect on gram+ (thick cell walled pathogens)
                      2) 80-90% renal excretion (unchanged)
                      3) typically: 25-30mg/kg loading dose, 15-20mg/kg maintenance; troughs of 10-20mg/kg or AUC/MIC>400
                          monitored after 3-5 doses
                      4) 
                equations:
                {
                    let k_cp_mean = kcp0
                            * x[7] // positive dependence on %t<C_eff, IF there is a blood infection.
                            * (1.0 + alpha_kcp/(1.0 + ((conc_cp_eff - vanc_conc)/kcp_slope).exp()));
                        dx[8] =  (k_cp_mean - x[8])/tau_kcp_reversion;
                        k_cp = x[8];
                    let a_kpc = x[5] / conc_peri_eff; // AUC_Dt/C_eff ... this needs to be sigmoidal, too ... w/eff-> k_pc0
                    let k_pc = a_kpc * k_pc0;
                } // conceptual development
            */ //
            let k_cp_mean = kcp0 *
                    drug_effect_on_k(alpha_kcp,p_inf_central,conc_central_eff,vanc_conc,kcp_slope);
                dx[8] =  (k_cp_mean - x[8])/tau_kcp_reversion;
            let k_cp = x[8];

            let kpc_slope = 3.5;
            let kpc_eff_e50 = kpc_e50;
            let k_pc_mean = kpc0 *
                    drug_effect_on_k(alpha_kpc,p_inf_periph,kpc_eff_e50,x[5]/conc_peri_eff,kpc_slope);
            let _k_pc_mean = if t > 40.0 {kpc0} else {alpha_kpc * kpc0};
                dx[9] =  (k_pc_mean - x[9])/tau_kpc_reversion;
            let k_pc = x[9];
            // */


            // user defined two-comp model
            let d_mg = rateiv[0] - (k_e + k_cp) * x[2] + k_pc * x[3];
            dx[2] =  d_mg;
            dx[3] = k_cp * x[2] - k_pc * x[3];

            // time> or time< MIC, running AUC, other stats
            dx[5] = (d_mg/vol) - x[5]/tau_auc; //  "/4.8;" // AUC(t-24) total  
            
            let tau_gt_conc_eff = tau_mic; // 
            let x_4 = if x[4] < 0.0 { 0.0 } else { x[4] };
            let x_6 = if x[6] < 0.0 { 0.0 } else { x[6] };
            if vanc_conc >= conc_peri_eff {
                dx[4] = 1.0 - x_4/tau_gt_conc_eff; // 33.6Hr is a leaky integrator w/5*tau ~ 1 week
                dx[6] = -1.0 * x_6/tau_gt_conc_eff; 
            } else {
                dx[4] = -1.0 * x_4/tau_gt_conc_eff;
                dx[6] = 1.0 - x_6/tau_gt_conc_eff;
            }  
            // let tau_gt_conc_eff = tau_mic; // all time constants on EC are the same
            let x_10 = if x[10] < 0.0 { 0.0 } else { x[10] };
            let x_12 = if x[12] < 0.0 { 0.0 } else { x[12] };
            if vanc_conc >= conc_central_eff {
                dx[10] = 1.0 - x_10/tau_gt_conc_eff;
                dx[12] = -1.0 * x_12/tau_gt_conc_eff; 
            } else {
                dx[10] = -1.0 * x_10/tau_gt_conc_eff;
                dx[12] = 1.0 - x_12/tau_gt_conc_eff;
            }
            //
            // if t > 24.0 { // this has to start from t=0, so initialize x_6 = 1.0e-8 and x_4 = 0.0 at t=0.
                if x[7] > 0.05 { // infection can't recover at this point
                    let _rel_time_lt_eff = x_6/(x_4 + x_6); // relative %time < concentration required _in_blood_ to have a _peripheral_ kill effect on pathogen
                    let rel_time_gt_eff = x_4/(x_4 + x_6);
                    let delta_eff = 1.0 /(1.0 + ((0.5 - rel_time_gt_eff)/0.08).exp()); // map to (0,1)
                    dx[7] =  delta_eff - x[7] / tau_p_periph_eff; // *** fix the potental division by zero ***
                } else {
                    dx[7] = - x[7] / tau_p_periph_eff; // if EFF resolved, do not allow to recur
                }
            // } else {
            //    dx[7] = 0.0;
            // }
                if x[11] > 0.05 {
                    let rel_time_lt_eff = x_12/(x_10 + x_12);
                    let _rel_time_gt_eff = x_10/(x_10 + x_12);
                    let delta_eff = 1.0/(1.0 + ((0.5 - rel_time_lt_eff)/0.08).exp());
                    dx[11] = delta_eff - x[11] / tau_p_cent_eff;
                } else {
                    dx[11] = - x[11] / tau_p_cent_eff;
                }
            // } // if t > 24
        },
        |p, d| {
            fetch_params!(p, _v0, _ke0, _kcp0, _kpc0 // non-infected state
                , _ke_vs_crcl // dependence on covariates
                , _p_periph_eff_0 // prob of infection effect on pk parameters at start of treatment
                , _tau_p_periph_eff // x_7(t=0); relative strength of infection
                , _p_cent_eff_0
                , _tau_p_cent_eff
                , _tau_mic // strength of infection (t) is ~ percent time above or below MIC
                , _conc_central_eff // drug E50 for k_e and k_cp
                , _alpha_ke , _ke_slope 
                , _alpha_kcp, _kcp_slope
                , _tau_auc // peripheral drug exposure is related to integrated difference in drug concentration between periphery and center
                , _conc_peri_eff // drug E50 for kpc
                , _alpha_kpc // kpc_slope=1.75
                , _kpc_e50
                , _tau_kel_reversion, _tau_kcp_reversion, _tau_kpc_reversion
                // , _ske
                // , svol
                );
            d[0] = 0.0; // ske * ke0;
            d[1] = 0.0; // svol * v0; // svol; // svol * v0;
            // the above increments MUST match the state increments of x
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, t, cov, x| {
            fetch_params!(p, v0, ke0, kcp0, kpc0 // non-infected state
                , ke_vs_crcl // dependence on covariates
                , p_periph_eff_0 // prob of infection effect on pk parameters at start of treatment
                , _tau_p_periph_eff // x_7(t=0); relative strength of infection
                , p_cent_eff_0
                , _tau_p_cent_eff
                , _tau_mic // strength of infection (t) is ~ percent time above or below MIC
                , conc_central_eff // drug E50 for k_e and k_cp
                , alpha_ke , ke_slope 
                , alpha_kcp, kcp_slope
                , _tau_auc // peripheral drug exposure is related to integrated difference in drug concentration between periphery and center
                , _conc_peri_eff // drug E50 for kpc; AUC/conc_peri_eff ... but because AUC=0 during init conc_peri_eff is not needed
                , alpha_kpc // kpc_slope=1.75
                , kpc_e50
                , _tau_kel_reversion, _tau_kcp_reversion, _tau_kpc_reversion
                // , _ske
                // , svol
                );
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t
            /*
            let normal_ke = Normal::new(ke0, ske*ke0).unwrap();
            x[0] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            */
            x[0] = ke0 * (1.0 + alpha_ke/(1.0 + (conc_central_eff/ke_slope).exp())) * ke_vs_crcl * crcl * (wt/70.0).powf(-0.25);
            // let normal_v = Normal::new(v0, svol*v0).unwrap();
            // x[1] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            x[1] = v0;
            x[2] = 0.0; // central compartment
            x[3] = 0.0; // peripheral compartment
            x[4] = 0.0; // time >= conc_peri_eff
            x[5] = 0.0; // (total AUC / tau_auc) if (tau_auc == 4.8) -> AUC(t-24Hr)
            x[6] = 1.0e-8; // time < or > conc_peri_eff
            x[7] = p_periph_eff_0; // -> x_6/(x_4 + x_6)
            x[8] = kcp0 *
                    drug_effect_on_k(alpha_kcp,x[7],conc_central_eff,0.0,kcp_slope); // k_cp_mean
            let kpc_slope = 1.75;
            let kpc_eff_e50 = kpc_e50; // units in AUC/C_eff
            x[9] = kpc0 *
                    drug_effect_on_k(alpha_kpc,x[7],kpc_eff_e50,0.0,kpc_slope);
            x[10] = 0.0;
            x[11] = p_cent_eff_0;
            x[12] = 1.0e-8;
        },
        |x, _p, t, cov, y| {
            // fetch_params!(p, _ke0, _kcp, _kpc, v0);
            fetch_cov!(cov,t,wt); // , crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = x[1] * (wt/70.0);

            y[0] = x[2]/vol;
        },
        (13, 1),
        1,
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
*/ // simulate a single point

let mut settings = Settings::new();

// First pass for the drug effect on both kcp and kpc for ID5 
/*
let params = Parameters::builder()
        .add("v0", 20.0, 170.0, true) //
        .add("ke0", 1.0e-4, 5.0, true)
        .add("kcp0", 1.0e-4, 5.0, true)
        .add("kpc0", 1.0e-4, 5.0, true)
        .add("ke_vs_crcl", 1.0e-2, 1.0, true)
        .add("p_periph_eff_0", 0.0, 1.0, true)
        .add("tau_p_periph_eff", 12.0, 36.0, true) 
        .add("p_cent_eff_0", 0.0, 1.0, true)
        .add("tau_p_cent_eff", 1.2, 4.8, true) 
        .add("tau_mic", 1.2, 12.0, true) // 4.8
        .add("conc_central_eff", LLQ, 2.5*MIC, true)
        .add("alpha_ke", -1.0, 0.0, true)//
        .add("ke_slope", 1.0e-3, 2.0, true) // 2.0
        .add("alpha_kcp", -1.0, 2.0, true)//
        .add("kcp_slope", 1.0e-3, 2.0, true) // 2.0
        .add("tau_auc", 1.2, 4.8, true) // 9.6
        .add("conc_peri_eff", LLQ, 2.5*MIC, true) // 2.5*MIC
        .add("alpha_kpc", -1.0, 0.0, true)//
        .add("kpc_E50", 25.0,200.0, true)
        .add("tau_kel_reversion", 1.2, 4.8, true) // 1.2,(4.8, 16.0)
        .add("tau_kcp_reversion", 1.2, 4.8, true) // 1.2,(4.8, 16.0)
        .add("tau_kpc_reversion", 1.2, 4.8, true) // 1.2,(4.8, 16.0)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 2.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();

    /* These values are promising. But has bias to lower predction than desirable.
v0                  67.31542 <- (20, 170)
ke0               0.01040365 <- (1e-4, 5.0)
kcp0                0.890867
kpc0               0.9596441
ke_vs_crcl        0.09530507
p_periph_eff_0     0.4765192
tau_p_periph_eff    13.96161
p_cent_eff_0       0.4965554
tau_p_cent_eff      3.404261
tau_mic             6.679888
conc_central_eff    24.84556
alpha_ke          -0.6692032
ke_slope            1.141451
alpha_kcp           1.889691
kcp_slope          0.2936495
tau_auc             2.009533
conc_peri_eff        14.4272
alpha_kpc         -0.3333533
kpc_E50             38.14727
tau_kel_reversion   1.556242
tau_kcp_reversion   3.267712
tau_kpc_reversion   3.823672
svol                    0.25  
     */
*/
// /* 
let params = Parameters::builder()
        // .add("v0", 67.3, 67.33084, true) //
        .add("v0", 20.0, 67.33084, true) //
        .add("ke0", 1.0e-2, 1.08073e-2, true)
        .add("kcp0", 8.9e-1, 8.91734e-1, true)
        .add("kpc0", 9.59e-1, 9.602881e-1, true)
        .add("ke_vs_crcl", 9.5e-2, 9.561014e-2, true)
        .add("p_periph_eff_0", 0.47, 0.4830382, true)
        .add("tau_p_periph_eff", 13.9, 14.92322, true) 
        .add("p_cent_eff_0", 0.49, 0.5031108, true)
        .add("tau_p_cent_eff", 3.4, 3.408522, true) 
        .add("tau_mic", 6.67, 6.89776, true) // 4.8
        .add("conc_central_eff", 24.8, 24.89112, true)
        .add("alpha_ke", -0.6784064, -0.66, true)//
        .add("ke_slope", 1.14, 1.142902, true) // 2.0
        .add("alpha_kcp", 1.8, 1.979382, true)//
        .add("kcp_slope", 0.29, 0.2972990, true) // 2.0
        .add("tau_auc", 2.0, 2.019066, true) // 9.6
        .add("conc_peri_eff", 14.0, 14.8344, true) // 2.5*MIC
        .add("alpha_kpc", -0.3367066, -0.33, true)//
        .add("kpc_E50", 38.0,38.29454, true)
        .add("tau_kel_reversion", 1.5, 1.612484, true) // 1.2,(4.8, 16.0)
        .add("tau_kcp_reversion", 3.2, 3.335424, true) // 1.2,(4.8, 16.0)
        .add("tau_kpc_reversion", 3.8, 3.847344, true) // 1.2,(4.8, 16.0)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 2.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();
   // */ // before playing w/the tau_..._reversions


/*
let params = Parameters::builder()
        .add("v0", 42.0, 46.5, true)
        .add("ke0", 7.3e-2, 7.9e-2, true)
        .add("kcp0", 1.25, 1.31, true)
        .add("kpc0", 3.2e-1, 3.6e-1, true)
        .add("ke_vs_crcl", 1.0e-2, 3.0e-2, true)
        .add("p_periph_eff_0", 0.0, 1.0, true) // x[7]
        .add("tau_p_periph_eff", 19.2, 67.2, true)
        .add("tau_mic", 1.2, 4.8, true) // 4.8
        .add("conc_central_eff", LLQ, 2.5*MIC, true)
        .add("alpha_ke", -1.0, 0.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("ke_slope", 1.0e-3, 2.0, true) // 2.0
        .add("alpha_kcp", -1.0, 0.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("kcp_slope", 1.0e-3, 2.0, true) // 2.0
        .add("tau_auc", 1.2, 9.6, true) // 9.6
        .add("conc_peri_eff", LLQ, 3.0*MIC, true) // 2.5*MIC
        .add("alpha_kpc", -1.0, 2.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("kpc_E50", 25.0,200.0, true)
        .add("tau_kel_reversion", 1.2, 4.80, true) // 1.2,(4.8, 16.0)
        .add("tau_kcp_reversion", 1.2, 9.6, true) // 1.2,(4.8, 16.0)
        .add("tau_kpc_reversion", 1.2, 4.80, true) // 1.2,(4.8, 16.0)
        // .add("ske", 0.0001, 0.5, true)
        .add("svol", 0.0001, 1.0, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();
*/ // IOS and SDE model for ID5

/* // ODE model (w/alot of overhead from IOS) for subject 5
let params = Parameters::builder()
        .add("v0", 20.0, 120.0, true)
        .add("ke0", 1.0e-5, 2.0, true)
        .add("kcp0", 1.0e-4, 2.0, true)
        .add("kpc0", 1.0e-4, 2.0, true)
        .add("ke_vs_crcl", 1.0e-4, 2.0, true)
        .add("p_periph_eff_0", 0.0, 1.0, true) // x[7]
        .add("tau_p_periph_eff", 19.2, 67.2, true)
        .add("tau_mic", 1.2, 4.8, true) // 4.8
        .add("conc_central_eff", LLQ, 2.5*MIC, true)
        // .add("alpha_ke", -1.0, 0.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("alpha_ke", -0.00001, 0.00001, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("ke_slope", 1.0e-3, 2.0, true) // 2.0
        // .add("alpha_kcp", -1.0, 0.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("alpha_kcp", -0.00001, 0.00001, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("kcp_slope", 1.0e-3, 2.0, true) // 2.0
        .add("tau_auc", 1.2, 9.6, true) // 9.6
        .add("conc_peri_eff", LLQ, 3.0*MIC, true) // 2.5*MIC
        // .add("alpha_kpc", -1.0, 2.0, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("alpha_kpc", -0.00001, 0.00001, true)// ke_mu in (1/2, 2x)ke0, w/E50=conc_peri_eff
        .add("kpc_E50", 25.0,200.0, true)
        .add("tau_kel_reversion", 1.2, 4.80, true) // 1.2,(4.8, 16.0)
        .add("tau_kcp_reversion", 1.2, 9.6, true) // 1.2,(4.8, 16.0)
        .add("tau_kpc_reversion", 1.2, 4.80, true) // 1.2,(4.8, 16.0)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 2.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();

*/
    settings.set_parameters(params);

    // settings.set_prior_sampler("sobol".to_string());
    // settings.set_prior_points(147000);
    // settings.set_prior_seed(347);

    settings.set_cycles(1000);
    // settings.set_error_poly((0.1, 0.075, -0.00165, 0.0)); // MN uses 0.1,0.15 ... CV%<0.1 is an acceptiable assay ... so 0, 0.1, ... is probably "right" to use for comparing SDE to ODE solutions
    settings.set_error_poly((0.0, 0.15, 0.0, 0.0));
    settings.set_error_type(ErrorType::Add);
    settings.set_error_value(2.0*LLQ);
    settings.set_idelta(1.0);

    // for ODE use this block:
   //  /*
        settings.set_output_path("examples/vpicu_for_grant_prop/output_ios_tmp"); // _arc_kel_kcp_kpc"); // THIS LINE OVERWRITES THIS DIRECTORY !!!
        settings.set_prior_sampler("sobol".to_string());
        settings.set_prior_points(146657);
        settings.set_prior_seed(347);
        // settings.set_prior(settings::Prior {
        //    sampler: "sobol".to_string(),
        //    points: 16384,
        //    seed: 347,
        //    file: None, // Some(String::from("examples/vpicu_for_grant_prop/output_ode/theta.csv")),
        // });
    // */

    // for SDE use this block (Verify AG is edited to expand only in dimensions of sigma: ___ YES ___):
    /*
        settings.set_output_path("examples/vpicu_for_grant_prop/output_ode_arc_kel_kcp_kpc"); // THIS LINE OVERWRITES THIS DIRECTORY !!!
        settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_ode_arc_kel_kcp_kpc/theta_add_sigma.csv")));
        // settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/prior_first_six.csv")));
        // settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_ode/theta_w_sigma.csv")));
        // settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_ode_arc/theta.csv")));
        //
        // to optimize ONLY the sigmas, edit src/routines/expansion/adaptive_grid.rs to only expand in the dimentions of sigma
        //
    */

    setup_log(&settings)?;
    let data = data::read_pmetrics("examples/vpicu_for_grant_prop/vpicu_5.csv")?; // subj1to6.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit().unwrap();
    result.write_outputs()?;

    Ok(())
}
