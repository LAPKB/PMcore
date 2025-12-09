use std::fs::File;
use std::path::Path;

use pharmsol::equation;
use plotly::common::Marker;
use plotly::{layout, Plot, Scatter};
use pmcore::prelude::settings::{Parameters, Prior, Settings};
use pmcore::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};
use std::io::Write;

fn model_ke_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke);

            // user defined
            dx[0] = -ke * x[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 20.0;
        },
        |x, _p, _t, _cov, y| {
            y[0] = x[0]; // v0 ~ N(1.0, sigma^2)
        },
        (1, 1),
    )
}

fn model_ke_v_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _v0);

            // user defined
            dx[0] = -ke0 * x[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 20.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, v0);
            y[0] = x[0]/v0;
        },
        (1, 1),
    )
}

fn model_ke() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke);

            // user defined
            dx[0] = -ke * x[0];
        },
        |_p, _d| {},
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 20.0;
        },
        |x, _p, _t, _cov, y| {
            y[0] = x[0];
        },
        (1, 1),
        1,
    )
}

fn model_ke_s1() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _s1);

            // mean reversion to ke0; ~ N(ke0,sigma=ke0/s1)
            dx[1] = -x[1] + ke0;
            let k_elim = x[1];

            // user defined
            dx[0] = -k_elim * x[0];
        },
        |p, d| {
            fetch_params!(p, ke0, s1);
            // d[1] is the sd for the Euler Maruyama update, i.e. ~ N(mu=0,sigma=ke0/s1) 
            d[1] = ke0 / s1; // during optimization s1 is a population r.v., but for simulation s1 is a constant.
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, s1);
            let k_elim0 = Normal::new(ke0, ke0/s1).unwrap();
            x[0] = 20.0;
            x[1] = k_elim0.sample(&mut rand::rng()); // ~ N(ke0,sigma=ke0/s1)
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, _s1);
            y[0] = x[0];
        },
        (2, 1),
        1,
    )
}

fn model_ke_v() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke, _v);

            // user defined
            dx[0] = -ke * x[0];
        },
        |_p, _d| {},
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 20.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
        1,
    )
}

fn model_ke_v_s1() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _v, _s1);

            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let k_elim = x[1];

            // user defined
            dx[0] = -k_elim * x[0];
        },
        |p, d| {
            fetch_params!(p, ke0, _v, s1);
            d[1] = ke0 / s1; // s1;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _v, s1);
            let k_elim0 = Normal::new(ke0, ke0/s1).unwrap();
            x[0] = 20.0;
            x[1] = k_elim0.sample(&mut rand::rng()); // ~ N(ke0,sigma=ke0/s1)
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v, _s1);
            y[0] = x[0] / v;
        },
        (2, 1),
        1,
    )
}

fn model_ke_v_s1_s2() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, v0, _s1, _s2);
            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // mean reversion to V
            dx[2] = -x[2] + v0;
            // let v = x[2]; // but vol is only used in out

            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, ke0, v0, s1, s2);
            d[1] = ke0 / s1;
            d[2] = v0 / s2;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, v0, s1, s2);
            let k_elim0 = Normal::new(ke0, ke0/s1).unwrap();
            let vol0 = Normal::new(v0, v0/s2).unwrap();
            x[0] = 20.0;
            x[1] = k_elim0.sample(&mut rand::rng()); // ~ N(ke0,sigma=ke0/s1)
            x[2] = vol0.sample(&mut rand::rng());
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _k0, _v0, _s1, _s2);
            let v = x[2];
            y[0] = x[0] / v;
        },
        (3, 1),
        1,
    )
}

fn settings_exp1() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke", 0.1, 3.0, false) // Range covers both normal distributions (0.5±0.05 and 1.5±0.15)
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn settings_exp2() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke0", 0.1, 3.0, false) // Base ke range
        .add("s1", 2.0, 20.0, false) // Diffusion parameter for ke
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn settings_exp3() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke", 0.1, 3.0, false)
        .add("v", 0.1, 3.0, false)
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn settings_exp4() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke0", 0.1, 3.0, false)
        .add("v", 0.1, 3.0, false) // Volume range based on V = 1 + 0.2*N(0,1)
        .add("s1", 2.0, 20.0, false) // Diffusion parameter for ke
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn settings_exp5() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke0", 0.1, 3.0, false)
        .add("v0", 0.1, 3.0, false) // Base volume range
        .add("s1", 1.0, 20.0, false) // Diffusion parameter for ke
        .add("s2", 1.0, 20.0, false) // Diffusion parameter for volume
        .build()
        .unwrap();

    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn settings_exp6() -> Settings {
    let mut settings = Settings::new();
    let params = Parameters::builder()
        .add("ke0", 0.1, 3.0, false)
        .add("v0", 0.1, 3.0, false) // Base volume range
        .build()
        .unwrap();
    settings.set_parameters(params);
    setup_common_settings(&mut settings)
}

fn setup_common_settings(settings: &mut Settings) -> Settings {
    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_error_value(0.0);
    settings.set_error_type(ErrorType::Add);
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 1000,
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
        .repeat(5, 0.2)
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
    1.0 + 0.2 * n3.sample(rng)
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
            let mut sb = data::Subject::builder(format!("id{}", i));
            for (t, point) in trajectory.iter().enumerate() {
                sb = sb.observation((t) as f64 * 0.2, point.prediction(), 0);
            }
            data.push(sb.build());
        });
    let data = data::Data::new(data);
    data.write_pmetrics(&File::create(Path::new(file_name)).unwrap());
}
const N_SAMPLES: usize = 100;

fn generate_data() {
    //k0 dist according to the paper F(K0) = 0.5*N(m_1,(s_1)^2) + 0.5*N(m_2,(s_2)^2)
    let m1 = 0.5;
    let s1 = 0.05;
    let m2 = 1.5;
    let s2 = 0.15;
    let n1 = Normal::new(m1, s1).unwrap();
    let n2 = Normal::new(m2, s2).unwrap();
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

    // plot.show();
    let mut file = File::create("examples/sde_paper/data/population.csv").unwrap();
    writeln!(file, "index,k0,v0").unwrap();
    for (i, (k0, v0)) in k0_pop.iter().zip(v_pop.iter()).enumerate() {
        writeln!(file, "{},{},{}", i, k0, v0).unwrap();
    }

    // Now, let's create the data files for each experiment

    // let s_ke0 = Normal::new(10.0, 2.0).unwrap(); // sigma_ke0 = ke0/s_ke0
    // let s_v0 = Normal::new(10.0, 2.0).unwrap();
    // let s1 = s_ke0.sample(&mut rand::rng()); .... this is incorrect; s1 and s2 do not have to be drawn from a distribution.

     //experiment 0: Random Ke, Volume ODE
    // uses the input file from experiment 1

    //experiment 1: Random Ke, ODE
    write_samples_to_file(
        "examples/sde_paper/data/experiment1.csv",
        k0_pop.clone(),
        None,
        10.0e128, // No stochastic component // ke ~ N(ke0,ke0/s1)
        10.0e128,
        model_ke(),
    );

    //experiment 2: Random Ke with random sigma_Ke
    write_samples_to_file(
        "examples/sde_paper/data/experiment2.csv",
        k0_pop.clone(),
        None,
        1.0, // s_ke0.sample(&mut rand::rng()), // see notes in model_ke_s1(), s1 and s2 are constants
        10.0e128,
        model_ke_s1(),
    );

    //experiment 3: Random Ke and random Volume
    write_samples_to_file(
        "examples/sde_paper/data/experiment3.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        10.0e128, // No stochastic component
        10.0e128,
        model_ke_v(),
    );

    //experiment 4: Random Ke, ske with random Volume
    write_samples_to_file(
        "examples/sde_paper/data/experiment4.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        1.0,
        10.0e128,
        model_ke_v(),
    );

    //experiment 5: Random Ke, Volume with their sigma parameters
    write_samples_to_file(
        "examples/sde_paper/data/experiment5.csv",
        k0_pop.clone(),
        Some(v_pop.clone()),
        1.0e3,
        1.0e128,
        model_ke_v_s1_s2(),
    );

    //experiment 6: Random Ke, Volume ODE
    // uses the input file from experiment 3

}

fn fit_experiment(experiment: usize) {
    // Experiments 0 and 6 are ODE models
    if experiment == 0 {
        let data = data::read_pmetrics("examples/sde_paper/data/experiment1.csv").unwrap();
        let eqn = model_ke_ode();
        let mut settings = settings_exp1();
        settings.set_output_path("examples/sde_paper/output/experiment0");
        settings.set_error_poly((0.0, 0.15, 0.0, 0.0));
        let mut problem = dispatch_algorithm(settings, eqn, data).unwrap();
        let result = problem.fit().unwrap();
        result.write_outputs().unwrap();
        return;
    }
    if experiment == 6 {
        let data = data::read_pmetrics("examples/sde_paper/data/experiment3.csv").unwrap(); // ke and v, w/sigma=0
        let eqn = model_ke_v_ode();
        let mut settings = settings_exp3();
        settings.set_output_path("examples/sde_paper/output/experiment6");
        settings.set_error_poly((0.0, 0.15, 0.0, 0.0));
        let mut problem = dispatch_algorithm(settings, eqn, data).unwrap();
        let result = problem.fit().unwrap();
        result.write_outputs().unwrap();
        return;
    }

    let data = data::read_pmetrics(&format!(
        "examples/sde_paper/data/experiment{}.csv",
        experiment
    ))
    .unwrap();

    let (eqn, settings) = match experiment {
        1 => (model_ke(), settings_exp1()),
        2 => (model_ke_s1(), settings_exp2()),
        3 => (model_ke_v(), settings_exp3()),
        4 => (model_ke_v_s1(), settings_exp4()),
        5 => (model_ke_v_s1_s2(), settings_exp5()),
        _ => panic!("Invalid experiment"),
    };

    let mut settings = settings;
    settings.set_output_path(&format!(
        "examples/sde_paper/output/experiment{}",
        experiment
    ));

    settings.set_error_poly(match experiment {
        1..=4 => (0.0, 0.15, 0.0, 0.0),
        5 => (1.0, 0.2, 0.0, 0.0),
        _ => panic!("Invalid experiment"),
    });

    let mut problem = dispatch_algorithm(settings, eqn, data).unwrap();
    let result = problem.fit().unwrap();
    result.write_outputs().unwrap();
}

fn main() {
    generate_data();
    for i in 0..=6 {
        fit_experiment(i);
    }
}
