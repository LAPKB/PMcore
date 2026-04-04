
use pharmsol::builder::SubjectBuilder;
use pmcore::prelude::*;

// use rand_distr::{Distribution, Normal};

const MIC:f64 = 10.0;
const LLQ:f64 = 4.0;


fn main()
    {

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
            fetch_params!(p, ke0, kcp0, kpc0, v0, alpha_ke, conc_peri_eff,tau_two2one, _p_inflammation_0); // , conc_peri_eff); // , alpha_ke, conc_peri_eff);
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t

            dx[1] = v0 - x[1]; // mean reverting sde
            let vol = x[1] * (wt/70.0);         
            let vanc_conc = x[2]/vol;
            // let k_e_mean = ke0 * (1.0 + alpha_ke/(1.0 + ((conc_peri_eff - vanc_conc)/2.0).exp())) ; //alpha_ke * ke0;
            let k_e_mean = ke0 * (1.0 + alpha_ke/(1.0 + ((conc_peri_eff - vanc_conc)/2.0).exp())) ; //alpha_ke * ke0;
            // dx[0] = ke0 - x[0]; // mean reverting to ke0
            dx[0] =  (k_e_mean - x[0])/1.0; // 168  336  504  672  840 1008 1176 1344 1512 1680
            let k_e = x[0] * (wt/70.0).powf(-0.25) * 0.64 * crcl / 120.9; // k_e ~ x[0] normalized to wt and crcl
            let k_cp = kcp0 * x[7] / (1.0 + x[5]); // 336.0); // make sure at T=0 k_cp doesn't go to infty
            let k_pc = kpc0;

            // user defined two-comp model
            let d_mg = rateiv[0] - (k_e + k_cp) * x[2] + k_pc * x[3];
            dx[2] =  d_mg;
            dx[3] = k_cp * x[2] - k_pc * x[3];

            // time>MIC, running AUC, other stats
            dx[5] = (d_mg/vol) - x[5]/tau_two2one; // "/4.8;" // AUC(t-24) total  
            if vanc_conc >= conc_peri_eff {
                dx[4] = 1.0 - x[4]/33.6; // 33.6Hr is a leaky integrator w/5*tau ~ 1 week
                dx[6] = -1.0 * x[6]/33.6; 
            } else {
                dx[4] = -1.0 * x[4]/33.6;
                dx[6] = 1.0 - x[6]/33.6;
            }         
            if t > 24.0 {
                dx[7] = x[6]/(x[4] + x[6]) - x[7] / 33.6;
            } else {
                dx[7] = 0.0;
            }        
        },
        |p, d| {
            // fetch_params!(p, ke0, _kcp, _kpc, v0, ske, svol);
            d[0] = 0.0; // ske * ke0;
            d[1] = 0.0; // svol * v0;
            // the above increments MUST match the state increments of x
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _kcp0, _kpc0, v0, alpha_ke, conc_peri_eff, _tau_two2one, p_inflammation_0);
            /*
            let normal_ke = Normal::new(ke0, ske*ke0).unwrap();
            x[0] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            let normal_v = Normal::new(v0, svol*v0).unwrap();
            x[1] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            */
            x[0] = ke0 * (1.0 + alpha_ke/(1.0 + (conc_peri_eff/2.0).exp()));
            x[1] = v0;
            x[2] = 0.0; // central compartment
            x[3] = 0.0; // peripheral compartment
            x[4] = 0.0; // time >= conc_peri_eff
            x[5] = 0.0; // (total AUC / 4.8) -> AUC(t-24Hr)
            x[6] = 0.0; // time < conc_peri_eff
            x[7] = p_inflammation_0;
        },
        |x, _p, t, cov, y| {
            // fetch_params!(p, _ke0, _kcp, _kpc, v0);
            fetch_cov!(cov,t,wt); // , crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = x[1] * (wt/70.0);

            y[0] = x[2]/vol;
        },
        (8, 1),
        1,
    );

    let supp_point = vec![0.0676, 0.00108, 2.06, 55.1, 4.50, 18.4, 229.0, 0.396]; 
    
    let subject = Subject::builder("999")
        .covariate("wt", 0.0, 8.600000381)
        .covariate("crcl", 0.0 ,81.45277461)
        .infusion(0., 150.0, 0, 1.0)
        .observation(0.0, -99.0,0)
        .repeat(420, 0.5)
        .build();

    let sim = eq.estimate_predictions(&subject, &supp_point); // simulator
    // sim is an array of predictions: time, obs

   for s in sim{
    println!("{},{}", s.time(), s.prediction());
   }

    }
