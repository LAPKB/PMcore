//
// example "vanco_vpicu_v002"
//

use std::fs::File;
// use std::intrinsics;
use std::path::Path;

use pharmsol::equation;
// use plotly::common::Marker;
// use plotly::{layout, Plot, Scatter};
use pmcore::prelude::settings::{Parameters, Prior, Settings};
use pmcore::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};
use std::io::Write;

/*
{
    // Here's what to do: fit the data two a two compartment IV model. Not sure how you want the model, but use this:
 
mod <- PM_model$new(
    pri = list(
        Ke0 = ab(0, 5),
        KCP = ab(0, 4),
        KPC = ab(0, 10),
        V0 = ab(0.1, 2)
    ),
 
    cov = list(
    wt = interp(),
    crcl = interp()
    ),
 
    sec = function(){
        Ke = Ke0 * (wt/70)^(-0.25) * (crcl/120) 
        V = V0 * (wt/70)
    },
 
    eqn = function(){
        two_comp_IV
    },
 
    out = function(){
        Y[1] = X[2]/V
    },
 
    err = list(
        additive(2, c(0.1,0.15,0,0))
    )
 
)
    Fit the model to the data without SDE. Model will be solved algebraically.
    You'll need Pmetrics >3.0.0 for that. Julian can help you install from r-universe.
    I need the PMout.Rdata object in the outputs at the end.
    Fit model to the data with SDE and Ke_sigma and V_sigma.
    I don't know what output you generate, but I will need concentrations every 1 hour for each subject,
    based on Ke_sigma and V_sigma set to 0, but other parameters what they were when you fit with SDE?
    Basically I want median concentrations based on posteriors for Ke0 and V0 without any IOV.
    Does that make sense?
    Thanks!

}
*/ // MN's instructions

// Variables that describe the experiment;
const S_FACTOR:f64 = 0.4;
static GENDATA:bool = false; // true to generate data, then false to optimize.

const N_PARTICLES:usize = 1 + 46 * (1 - GENDATA as usize);

// println!("S_FACTOR={};GENDATA={};NPARTICLES={}",S_FACTOR,GENDATA,N_PARTICLES);

// const N_PARTICLES:usize = 80; // use 1 to generate data, and use much more (80-ish) to optimize
//
const M1:f64 = 0.75_f64;
const S1:f64 = 0.15_f64;
const M2:f64 = 1.5_f64;
const KE_OVERLAP: f64 = 3.0; // bimodal distribution overlaps this number of sd away from respective mean
static S2:f64 = (M2 - (M1 + KE_OVERLAP*S1))/KE_OVERLAP;
static VAR_KE: f64 = S1*S1 + S2*S2; // i.e. make sigma average of s1 and s2
// V is N(1.0, 0.2^2)

fn model_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            // automatically defined
            fetch_params!(p, ke0, kcp, kpc, _v0);
            fetch_cov!(cov,t,wt, crcl); // automatically interpolates, so you need t

            let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            // let V = V0 * (wt/70)

            // user defined
            dx[0] = rateiv[0] - (k_e + kcp) * x[0] + kpc * x[1];
            dx[1] = kcp * x[0] - kpc * x[1]
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 0.0;
            x[1] = 0.0; // pretty sure these are not necesary, but I like it.
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, _kcp, _kpc, v0);
            y[0] = x[0]/v0;
        },
        (2, 1), // extra output equations are used in SDE to collect statistics
    )
}

// SO
fn model_sde_no_7() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, v0); // , _s1, _s2);
            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // mean reversion to V
            dx[2] = -x[2] + v0;
            // let v = x[2]; // the r.v. v is only used in output.

            // SDE
            dx[0] = -ke * x[0];
            // ODE -- centering function
            dx[3] = -ke0 * x[3];

            // running statistics on the state
            dx[4] = (x[1] - ke0).powf(2.0)/ke0; // Chi^2
            dx[5] = (x[2] - v0).powf(2.0)/v0; // Chi^2
        },
        |p, d| {
            fetch_params!(p, _ke0, _v0, s1, s2);
            d[1] = s1;
            d[2] = s2;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, k0, v0, s1, s2); // sigma are optimized
            x[0] = 20.0;
            let normal_ke = Normal::new(k0, s1).unwrap();
            x[1] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            let normal_v = Normal::new(v0, s2).unwrap();
            x[2] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            x[3] = 20.0; // ODE -- centerring function
            x[4] = 0.0; // running chi^2 on ke and vol
            x[5] = 0.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _k0, v0); // , _s1, _s2);
            let v = x[2];
            y[0] = x[0] / v; // SDE 
            y[1] = x[3] / v0; // x[3] / v0; // ODE center ... or x[0] / v; // SDE
            y[2] = x[4]; // chi^2 on elimination
            y[3] = x[5]; // chi^2 on volume
        },
        (6, 4),
    N_PARTICLES,
    )
}

// StO
fn model_sde_fixed_sigma() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, v0); // , _s1, _s2);
            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // mean reversion to V
            dx[2] = -x[2] + v0;
            // let v = x[2]; // the r.v. v is only used in output.

            // SDE
            dx[0] = -ke * x[0];
            // ODE -- centering function
            dx[3] = -ke0 * x[3];

            // running statistics on the state
            dx[4] = (x[1] - ke0).powf(2.0)/ke0; // Chi^2
            dx[5] = (x[2] - v0).powf(2.0)/v0; // Chi^2
        },
        |_p, d| {
            // let m1 = 0.75_f64;
            // let s1 = 0.15_f64;
            // let m2 = 1.5_f64;
            // let sigma_ke = (S1.powf(2.0)+ ((M2 - (M1 + KE_OVERLAP*S1))/KE_OVERLAP).powf(2.0)).powf(0.5);
            let sigma_ke = VAR_KE.powf(0.5);
            d[1] = S_FACTOR * sigma_ke;
            d[2] = S_FACTOR * 0.2;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            /*
            fetch_params!(p, ke0, v0); // sigma are 0 and not optimized
            x[0] = 20.0;
            x[1] = ke0; // mean reversion: dx[1] = (-x[1] + ke0) = 0 => (ke = x[1]) => ke = k0
            x[2] = v0;
            x[3] = 20.0; // ODE -- centering function
            */

            // /*
            fetch_params!(p, k0, v0); // sigma are fixed and not optimized
            // let m1 = 0.75_f64;
            // let s1 = 0.15_f64;
            // let m2 = 1.5_f64;
            // let sigma_ke = (S1.powf(2.0)+ ((M2 - (M1 + KE_OVERLAP*s1))/KE_OVERLAP).powf(2.0)).powf(0.5);
            let sigma_ke = VAR_KE.powf(0.5);
            x[0] = 20.0;
            let normal_ke = Normal::new(k0, S_FACTOR*sigma_ke).unwrap();
            x[1] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            let normal_v = Normal::new(v0, S_FACTOR*0.2).unwrap();
            x[2] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            x[3] = 20.0; // ODE -- centering function
            // */

            x[4] = 0.0; // running chi^2 on ke and vol
            x[5] = 0.0;

            // println!("{}")
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _k0, v0); // , _s1, _s2);
            let v = x[2];
            y[0] = x[0] / v; // SDE 
            y[1] = x[3] / v0; // x[3] / v0; // ODE center ... or x[0] / v; // SDE
            y[2] = x[4]; // chi^2 on elimination
            y[3] = x[5]; // chi^2 on volume
        },
        (6, 4),
        N_PARTICLES,
    )
}

// SS
fn model_sde_no_9() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, v0); // , _s1, _s2);
            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // mean reversion to V
            dx[2] = -x[2] + v0;
            // let v = x[2]; // the r.v. v is only used in output.

            // SDE
            dx[0] = -ke * x[0];
            // ODE -- centering function
            dx[3] = -ke0 * x[3];

            // running statistics on the state
            dx[4] = (x[1] - ke0).powf(2.0)/ke0; // Chi^2
            dx[5] = (x[2] - v0).powf(2.0)/v0; // Chi^2
        },
        |p, d| {
            fetch_params!(p, _ke0, _v0, s1, s2);
            d[1] = s1;
            d[2] = s2;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, k0, v0, s1, s2); // sigma are optimized
            x[0] = 20.0;
            let normal_ke = Normal::new(k0, s1).unwrap();
            x[1] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            let normal_v = Normal::new(v0, s2).unwrap();
            x[2] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            x[3] = 20.0; // ODE -- centerring function

            x[4] = 0.0; // running chi^2 on ke and vol
            x[5] = 0.0;

            // println!("{}")
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _k0, _v0); // , _s1, _s2);
            let v = x[2];
            y[0] = x[0] / v; // SDE 
            y[1] = x[0] / v; // x[3] / v0; // ODE center ... or x[0] / v; // SDE
            y[2] = x[4]; // chi^2 on elimination
            y[3] = x[5]; // chi^2 on volume
        },
        (6, 4),
        N_PARTICLES,
    )
}

// StSt
fn model_sde_no_10() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, v0); // , _s1, _s2);
            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // mean reversion to V
            dx[2] = -x[2] + v0;
            // let v = x[2]; // the r.v. v is only used in output.

            // SDE
            dx[0] = -ke * x[0];
            // ODE -- centering function
            dx[3] = -ke0 * x[3];

            // running statistics on the state
            dx[4] = (x[1] - ke0).powf(2.0)/ke0; // Chi^2
            dx[5] = (x[2] - v0).powf(2.0)/v0; // Chi^2
        },
        |_p, d| {
            // let m1 = 0.75_f64;
            // let s1 = 0.15_f64;
            // let m2 = 1.5_f64;
            // let sigma_ke = (S1.powf(2.0)+ ((M2 - (M1 + KE_OVERLAP*s1))/KE_OVERLAP).powf(2.0)).powf(0.5);
            let sigma_ke = VAR_KE.powf(0.5);
            d[1] = S_FACTOR * sigma_ke;
            d[2] = S_FACTOR * 0.2;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, k0, v0); // sigma are fixed and not optimized
            // let m1 = 0.75_f64;
            // let s1 = 0.15_f64;
            // let m2 = 1.5_f64;
            // let sigma_ke = (S1.powf(2.0)+ ((M2 - (M1 + KE_OVERLAP*s1))/KE_OVERLAP).powf(2.0)).powf(0.5);
            let sigma_ke = VAR_KE.powf(0.5);
            x[0] = 20.0;
            let normal_ke = Normal::new(k0, S_FACTOR*sigma_ke).unwrap();
            x[1] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            let normal_v = Normal::new(v0, S_FACTOR*0.2).unwrap();
            x[2] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            x[3] = 20.0; // ODE -- centering function

            x[4] = 0.0; // running chi^2 on ke and vol
            x[5] = 0.0;
        },
        |x, _p, _t, _cov, y| {
            // fetch_params!(p, _k0, v0); // , _s1, _s2);
            let v = x[2];
            y[0] = x[0] / v; // SDE 
            y[1] = x[0] / v; // x[3] / v0; // ODE center ... or x[0] / v; // SDE
            y[2] = x[4]; // chi^2 on elimination
            y[3] = x[5]; // chi^2 on volume
        },
        (6, 4),
        N_PARTICLES,
    )
}

// S0S0
fn model_sde_eq_0() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ke0, v0); // , _s1, _s2);
            // mean reversion to ke0 -- but parameter should NEVER move away from ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // mean reversion to V -- same
            dx[2] = -x[2] + v0;
            // let v = x[2]; // the r.v. v is only used in output.

            // SDE
            dx[0] = -ke * x[0];
            // ODE -- centering function
            dx[3] = -ke0 * x[3];

            // running statistics on the state
            dx[4] = (x[1] - ke0).powf(2.0)/ke0; // Chi^2
            dx[5] = (x[2] - v0).powf(2.0)/v0; // Chi^2
        },
        |_p, d| {
            d[1] = 0.0;
            d[2] = 0.0; // s = 0.0
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, v0); // sigma are 0 and not optimized
            x[0] = 20.0;
            x[1] = ke0; // mean reversion: dx[1] = (-x[1] + ke0) = 0 => (ke = x[1]) => ke = k0
            x[2] = v0;
            x[3] = 20.0; // ODE -- centering function   
        },
        |x, _p, _t, _cov, y| {
            // fetch_params!(p, _k0, _v0); // , _s1, _s2);
            let v = x[2];
            /*
                Because SDE magnitude is 0, the SDE should be equal to the ODE during the
                entire duration of the optimization, from initialization through convergence.
            */
            y[0] = x[0] / v; // SDE 
            y[1] = x[0] / v; // x[3] / v0; // ODE center ... or x[0] / v; // SDE
            y[2] = x[4]; // chi^2 on elimination
            y[3] = x[5]; // chi^2 on volume
        },
        (6, 4),
        N_PARTICLES,
    )
}

fn settings_optimize_sigma() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke0", 0.0001, M2 + 5.0 * VAR_KE.sqrt(), false)
        .add("v0", 0.0001, 2.0, false) // Base volume range
        .add("s1", 0.0, 3.0 * VAR_KE.sqrt(), false) // Diffusion parameter for ke
        .add("s2", 0.0, 0.6, false) // Diffusion parameter for volume
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn settings_fixed_sigma() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke0", 0.0001, 1.5 + 5.0 * VAR_KE.sqrt(), false)
        .add("v0", 0.0001, 1.0+5.0*0.2, false) // Base volume range
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn setup_common_settings(settings: &mut Settings) -> Settings {
    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_error_value(2.0); // was set to 0.0
    settings.set_error_type(ErrorType::Add);
    settings.set_error_poly((0.1, 0.15, 0.0, 0.0));
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 8192,
        seed: 347,
        file: None,
    });
    settings.set_output_write(true);
    settings.set_log_level(settings::LogLevel::DEBUG);
    setup_log(&settings).unwrap();
    settings.clone()
}

fn subject() -> Subject {
    data::Subject::builder("id1")
        // .bolus(0.0, 20.0, 0)
        .observation(0.0, -1.0, 0)
        // .repeat(5, 0.2)
        .observation(0.0, -1.0, 1)
        .observation(0.2, -1.0, 0)
        .observation(0.2, -1.0, 1)
        .observation(0.225, -1.0, 0)
        .observation(0.225, -1.0, 1)
        .observation(0.25, -1.0, 0)
        .observation(0.25, -1.0, 1)
        .observation(0.275, -1.0, 0)
        .observation(0.275, -1.0, 1)
        .observation(0.3, -1.0, 0)
        .observation(0.3, -1.0, 1)
        .observation(0.75, -1.0, 0)
        .observation(0.75, -1.0, 1)
        .observation(0.775, -1.0, 0)
        .observation(0.775, -1.0, 1)
        .observation(0.8, -1.0, 0)
        .observation(0.8, -1.0, 1)
        .observation(0.825, -1.0, 0)
        .observation(0.825, -1.0, 1)
        .observation(1.0, -1.0, 0)
        .observation(1.0, -1.0, 1)
        .observation(1.0, -1.0, 2)
        .observation(1.0, -1.0, 3)
        // In pharmsol, likelihood.rs, likelihood of outputs 2 and 3 = 1.0
        // .observation_with_error(1.0, -1.0, 2, (1.0,1.0)e, true)
        // .observation_with_error(0.0, -1.0, 3, None, true)
        // .observation_with_error(0.0, -1.0, 3, None, true)
        .build()
}
fn sample_k0(rng: &mut StdRng, n1: Normal<f64>, n2: Normal<f64>) -> f64 {
    let weights = [0.5, 0.5];
    let dist = WeightedIndex::new(&weights).unwrap();

    let component = dist.sample(rng);
    match component {
        0 => n1.sample(rng),
        1 => n2.sample(rng),
        _ => panic!("Invalid component"),
    }
}
fn sample_v(rng: &mut StdRng, n3: Normal<f64>) -> f64 {
    {
        (1.0 + 0.2 * n3.sample(rng)).abs()
    } // n3 ~ N(0,1)
}

fn write_samples_to_file(
    file_name: &str,
    k0_pop: Vec<f64>,
    v_pop: Option<Vec<f64>>,
    s1: f64,
    s2: f64,
    sde: equation::SDE,
) {
    let subject = subject();
    let mut data = Vec::new();
    let v_pop = v_pop.unwrap_or_else(|| vec![1.0; N_SAMPLES]);
    k0_pop
        .iter()
        .enumerate()
        .zip(v_pop.iter())
        .for_each(|((i, k0), v)| {
            let spp = vec![*k0, *v, s1, s2];
            let trajectories = sde.estimate_predictions(&subject, &spp);
            let trajectory = trajectories.row(0);
            // for (t, point) in trajectory.iter().enumerate() {
                let mut sb = data::Subject::builder(format!("id{}", i));
                // let mut sb2 = data::Subject::builder(format!("id{}", i));
                for (_t, point) in trajectory.iter().enumerate() {
                    // for eqn in 0..1 {
                        sb = sb.observation(point.time(), point.prediction(), point.outeq());
                        // println!("{} {} {}",i,point.prediction(),point.outeq());
                    // }
                    // sb = sb.observation((t) as f64 * 0.2, point.prediction(), 0);
                    // sb2 = sb2.observation((t) as f64 * 0.2, point.prediction(), 1);
                }
                data.push(sb.build());
                // data.push(sb2.build());
            // } // sb2 has same output as sb.
        });
    let data = data::Data::new(data);
    data.write_pmetrics(&File::create(Path::new(file_name)).unwrap());
}
const N_SAMPLES: usize = 100;

fn generate_data() {
    //k0 dist according to the paper F(K0) = 0.5*N(m_1,(s_1)^2) + 0.5*N(m_2,(s_2)^2)
    let n1 = Normal::new(M1, S1).unwrap();
    let n2 = Normal::new(M2, S2).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(347);
    let mut k0_pop = Vec::new();
    let mut v_pop = Vec::new();

    //V dist according to the paper F(V) = 1+0.2*N(0,1)
    let n3 = Normal::new(0.0, 1.0).unwrap();

    for _ in 0..N_SAMPLES {
        k0_pop.push(sample_k0(&mut rng, n1, n2));
        v_pop.push(sample_v(&mut rng, n3));
    }

    // plot
/*
    let trace = Scatter::new(k0_pop.clone(), v_pop.clone())
        .mode(plotly::common::Mode::Markers)
        .name("Population")
        .marker(Marker::new().size(8).opacity(0.6));

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        layout::Layout::new()
            .title("Population Distribution")
            .x_axis(layout::Axis::new().title("k0"))
            .y_axis(layout::Axis::new().title("V")),
    );
    plot.show();
*/
    let mut file = File::create("examples/sde_paper/data/population.csv").unwrap();
    writeln!(file, "index,k0,v").unwrap();
    for (i, (k0, v)) in k0_pop.iter().zip(v_pop.iter()).enumerate() {
        writeln!(file, "{},{},{}", i, k0, v).unwrap();
    }

    // Now, let's create the data files for each experiment
    //experiment 1: Random Ke, ODE
    write_samples_to_file(
        "examples/sde_paper/data/experiment1.csv",
        k0_pop.clone(),
        None,
        0.0, // No stochastic component
        0.0,
        model_ke(),
    );

    //experiment 2: Random Ke with random sigma_Ke
    write_samples_to_file(
        "examples/sde_paper/data/experiment2.csv",
        k0_pop.clone(),
        None,
        1.0e-3,
        0.0,
        model_ke_s1(),
    );

    //experiment 3: Random Ke and random Volume
    write_samples_to_file(
        "examples/sde_paper/data/experiment3.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        0.0, // No stochastic component
        0.0,
        model_ke_v(),
    );

    //experiment 4: Random Ke, ske with random Volume
    write_samples_to_file(
        "examples/sde_paper/data/experiment4.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        1.0,
        0.0,
        model_ke_v(),
    );

    //experiment 5: Random Ke, Volume with their sigma parameters
    write_samples_to_file(
        "examples/sde_paper/data/experiment5.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        0.0, // control
        // 0.01798196, // 1/16
        // 0.03596393, // 1/8
        // 0.07192785, // 1/4
        // 0.1438557, // x0.5
        // 0.2877113, // 0.2877113 = sqrt(ke_s1^2 + ke_s2^2)
        // 0.5754226, // x2
        // 1.150845, // x4
        // 1.438557, // x5
        // 1.726268, // x6
        // 2.013979, // x7
        // 2.157835, // x7.5
        // 2.30169, // x8
        // 2.445546, // x8.5
        // 2.589402, // x9
        // 2.733257, // x9.5
        // 2.877113, // x10
        // 4.027958, // x14 -- ODE fails at Burke
        // 5.754226, // x20
        0.0, // control
        // 0.0125, // 1/16
        // 0.025, // 1/8
        // 0.05, // 1/4
        // 0.1, // 1/2
        // 0.2, // population sigma
        // 0.3,
        // 0.4, // x2
        model_ke_v_s1_s2(),
    );

    //experiment 6:
      //edited experiment 5: Random Ke, Volume with their sigma parameters
    write_samples_to_file(
        "examples/sde_paper/data/experiment6.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        // 0.0, // control
        0.01798196, // 1/16
        // 0.03596393, // 1/8
        // 0.07192785, // 1/4
        // 0.1438557, // x0.5
        // 0.2877113,  // 0.2877113 = sqrt(ke_s1^2 + ke_s2^2)
        // 0.5754226, // x2
        // 1.150845, // x4
        // 1.438557, // x5
        // 1.726268, // x6
        // 2.013979, // x7
        // 2.157835, // x7.5
        // 2.30169, // x8
        // 2.445546, // x8.5
        // 2.589402, // x9
        // 2.733257, // x9.5
        // 2.877113, // x10
        // 4.027958, // x14 -- ODE fails at Burke
        // 5.754226, // x20
        // 0.0, // control
        0.0125, // 1/16
        // 0.025, // 1/8
        // 0.05, // 1/4
        // 0.1, // 1/2
        // 0.2, // population sigma
        // 0.3,
        // 0.4, // x2
        model_sde_no_6(),
    );

    //experiment 7:
    write_samples_to_file(
        "examples/sde_paper/data/experiment7.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        // 0.0, // control
        S_FACTOR * VAR_KE.powf(0.5), // ... but x1/16 does pretty much nothing
        // 0.0, // control
        S_FACTOR * 0.2, // ... but x1/16 does pretty much nothing
        model_sde_no_7(),
    );
    //experiment 8:
    // Fixed parameter version of experiment 7.
    write_samples_to_file(
        "examples/sde_paper/data/experiment8.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        // 0.0, // control
        S_FACTOR * VAR_KE.powf(0.5), // ... but x1/16 does pretty much nothing
        // 0.0, // control
        S_FACTOR * 0.2, // ... but x1/16 does pretty much nothing
        model_sde_fixed_sigma(),
    );
    //experiment 9:
    write_samples_to_file(
        "examples/sde_paper/data/experiment9.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        // 0.0, // control
        S_FACTOR * VAR_KE.powf(0.5), // ... but x1/16 does pretty much nothing
        // 0.0, // control
        S_FACTOR * 0.2, // ... but x1/16 does pretty much nothing
        model_sde_no_9(),
    );
    //experiment 10:
    write_samples_to_file(
        "examples/sde_paper/data/experiment10.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        // 0.0, // control
        S_FACTOR * VAR_KE.powf(0.5), // ... but x1/16 does pretty much nothing
        // 0.0, // control
        S_FACTOR * 0.2, // ... but x1/16 does pretty much nothing
        model_sde_no_10(),
    );
    //experiment 11:
    write_samples_to_file(
        "examples/sde_paper/data/experiment11.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        0.0,
        0.0, // control
        model_sde_eq_0(),
    );

}

fn fit_experiment(experiment: usize) {
    let data = data::read_pmetrics("examples/vanco_vpicu_v02/data/vpicu.csv").unwrap();
   
    // Experiment 0 is the ODE model
    if experiment == 0 {
        let eqn = model_ode();
        let mut settings = settings_exp1();
        settings.set_output_path("examples/sde_paper/output/experiment0");
        // settings.set_error_poly((1.0, 0.05, 0.0, 0.0));// done in common
        let mut problem = dispatch_algorithm(settings, eqn, data).unwrap();
        let result = problem.fit().unwrap();
        result.write_outputs().unwrap();
        return;
    }
    // let data = data::read_pmetrics(&format!(
    //     "examples/sde_paper/data/experiment{}.csv",
    //     7 // experiment // 5 to have stochastic r.v.s
    // ))
    // .unwrap();

    let (eqn, settings) = match experiment {
        1 => (model_ke(), settings_exp1()),
        2 => (model_ke_s1(), settings_exp2()),
        3 => (model_ke_v(), settings_exp3()),
        4 => (model_ke_v_s1(), settings_exp4()),
        5 => (model_ke_v_s1_s2(), settings_exp5()),
        6 => (model_sde_no_6(), settings_exp6()),
        7 => (model_sde_no_7(), settings_optimize_sigma()), // SO
        8 => (model_sde_fixed_sigma(), settings_fixed_sigma()), // StO
        9 => (model_sde_no_9(), settings_optimize_sigma()), // SS
        10 => (model_sde_no_10(), settings_fixed_sigma()), // StSt
        11 => (model_sde_eq_0(), settings_fixed_sigma()), // S0S0
        _ => panic!("Invalid experiment"),
    };

    let mut settings = settings;
    settings.set_output_path(&format!(
        "examples/sde_paper/output/experiment{}",
        experiment
    ));

    settings.set_error_poly(match experiment {
        1..=4 => (0.0, 0.01, 0.0, 0.0),
        5 => (0.5, 0.02, 0.0, 0.0),
        6 => (1.0, 0.36, 0.0, 0.0), // C0 in 0.4, 0.8, 1.0, 1.2, 1.6 ... min obs is approx. 2, 1/2 = 1 (our usual protocol)
        7..=11 => (1.0, 0.05, 0.0, 0.0), // optimizing sigma
        // 8 => (1.0, 0.05, 0.0, 0.0), // fixed sigma
        _ => panic!("Invalid experiment"),
    });

    let mut problem = dispatch_algorithm(settings, eqn, data).unwrap();
    let result = problem.fit().unwrap();
    result.write_outputs().unwrap();
}

fn main() {
    // 1) Comment out fit_experiment(), then Generate data.
    if GENDATA {
        generate_data();
    // 2) Next, remove the ODE and statistical outputs in R; and set Y[1] = Y[0], i.e. both outputs are SDE
    } else {
        fit_experiment(0); // ODE control
        fit_experiment(1); // SDE experiments
    }
    /*
    for i in 0..=2 { // 0..=5 runs all of them
        println!("Start experiment {}",i);
        fit_experiment(i);
        // cp ~wyamada/src/lapk/PMcore/examples/sde_paper/output/data/experiment___.csv 
        //    ~wyamada/src/lapk/PMcore/examples/sde_paper/output/output/experiment___/
    }
    */
}
