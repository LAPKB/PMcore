use std::fs::File;
use std::io::Write;
use std::path::Path;

use pharmsol::equation;
use pmcore::prelude::*;

// For data generation only
use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};

fn model_ke_ode() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _v0);

            // user defined
            dx[0] = -ke0 * x[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 20.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, v0);
            y[0] = x[0] / v0;
        },
        (1, 1),
    )
}

fn model_ke() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _v0);

            // user defined
            dx[0] = -ke0 * x[0];
        },
        |_p, _d| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 20.0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, v0);
            y[0] = x[0] / v0;
        },
        (1, 1),
        101,
    )
}

fn model_ke_s1() -> equation::SDE {
    equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _v0, _s1);

            // mean reversion to ke0
            dx[1] = -x[1] + ke0;
            let ke = x[1];

            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0, _v0, s1);
            d[1] = s1;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _v0, _s1);
            x[0] = 20.0;
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, v0, _s1);
            y[0] = x[0] / v0;
        },
        (2, 1),
        101,
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
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
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
            let ke = x[1];

            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke, _v, s1);
            d[1] = s1;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _v, _s1);
            x[0] = 20.0;
            x[1] = ke0;
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
            // let v = x[2];
            // user defined
            dx[0] = -ke * x[0];
            // ODE
            dx[3] = -ke * x[3];
        },
        |p, d| {
            fetch_params!(p, _ke0, _v0, s1, s2);
            d[1] = s1;
            d[2] = s2;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, k0, v0, _s1, _s2);
            x[0] = 20.0;
            x[1] = k0;
            x[2] = v0;
            x[3] = 20.0; // ODE -- Like = ODE + SDE
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _k0, v0, _s1, _s2);
            let v = x[2];
            y[0] = x[0] / v;
            y[1] = x[3] / v0;
        },
        (4, 2),
        47,
    )
}

fn settings_exp1() -> Settings {
    let params = Parameters::new()
        .add("ke0", 2.5e-2, 1.5 + 5.0 * (1.5 - (0.5 + 3.0 * 0.05)) / 3.0)
        .add("v0", 0.0001, 2.0);

    setup_common_settings(params)
}

fn settings_exp2() -> Settings {
    let params = Parameters::new()
        .add("ke0", 1.0e-7, 3.5)
        .add("v0", 0.2, 1.8)
        .add("s1", 0.01, 2.0);

    setup_common_settings(params)
}

fn settings_exp3() -> Settings {
    let params = Parameters::new().add("ke", 0.1, 3.0).add("v", 0.1, 3.0);

    setup_common_settings(params)
}

fn settings_exp4() -> Settings {
    let params = Parameters::new()
        .add("ke0", 0.1, 3.0)
        .add("v", 0.1, 3.0)
        .add("s1", 0.0000000001, 2.0);

    setup_common_settings(params)
}

fn settings_exp5() -> Settings {
    let params = Parameters::new()
        .add("ke0", 2.5e-2, 1.5 + 5.0 * (1.5 - (0.5 + 3.0 * 0.05)) / 3.0)
        .add("v0", 0.0001, 2.0)
        .add("s1", 0.0025, 5.4)
        .add("s2", 0.01, 0.4);

    setup_common_settings(params)
}

fn setup_common_settings(params: Parameters) -> Settings {
    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.0, 0.0, 0.0), 1.0, None);
    let ems = ErrorModels::new().add(0, em).unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_prior(Prior::sobol(4851, 347));
    settings.set_write_logs(true);
    settings.set_log_level(LogLevel::DEBUG);
    settings.initialize_logs().unwrap();
    settings
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
    {
        (1.0_f64 + 0.2 * n3.sample(rng)).abs()
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
            let trajectories = sde.estimate_predictions(&subject, &spp).unwrap();
            let trajectory = trajectories.row(0);
            let mut sb = data::Subject::builder(format!("id{}", i));
            for (t, point) in trajectory.iter().enumerate() {
                sb = sb.observation((t) as f64 * 0.2, point.prediction(), 0);
            }
            data.push(sb.build());
        });
    let data = data::Data::new(data);
    let _ = data.write_pmetrics(&File::create(Path::new(file_name)).unwrap());
}
const N_SAMPLES: usize = 100;

fn generate_data() {
    //k0 dist according to the paper F(K0) = 0.5*N(m_1,(s_1)^2) + 0.5*N(m_2,(s_2)^2)
    let m1 = 0.5;
    let s1 = 0.05;
    let m2 = 1.50;
    let s2 = (m2 - (m1 + 3.0 * s1)) / 3.0; // let s2 = 0.15;
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
    */
    // plot.show();
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
        0.5346338, // (ke_s1^2 + ke_s2^2)^0.5 = 0.5346338
        0.2,       // sigma_vol is hard coded to 0.2 => 20%CV
        model_ke_v_s1_s2(),
    );
}

fn fit_experiment(experiment: usize) {
    // Experiment 0 is the ODE model // note: optimize on exp#5, which has stochastic r.v.s

    if experiment == 0 {
        let data = data::read_pmetrics("examples/sde_paper/data/experiment5.csv").unwrap();
        let eqn = model_ke_ode();

        // Rebuild settings with different error model
        let params = Parameters::new()
            .add("ke0", 2.5e-2, 1.5 + 5.0 * (1.5 - (0.5 + 3.0 * 0.05)) / 3.0)
            .add("v0", 0.0001, 2.0);
        let em = ErrorModel::proportional(ErrorPoly::new(0.0, 0.01, 0.0, 0.0), 1.0, None);
        let ems = ErrorModels::new().add(0, em).unwrap();
        let mut settings = Settings::builder()
            .set_algorithm(Algorithm::NPAG)
            .set_parameters(params)
            .set_error_models(ems)
            .build();
        settings.set_cycles(usize::MAX);
        settings.set_cache(true);
        settings.set_prior(Prior::sobol(4851, 347));
        settings.set_output_path("examples/sde_paper/output/experiment0");
        settings.set_write_logs(true);
        settings.set_log_level(LogLevel::DEBUG);
        settings.initialize_logs().unwrap();

        let mut problem = dispatch_algorithm(settings, eqn, data).unwrap();
        let result = problem.fit().unwrap();
        result.write_outputs().unwrap();
        return;
    }
    let data = data::read_pmetrics(&format!(
        "examples/sde_paper/data/experiment{}.csv",
        experiment // 5 to have stochastic r.v.s
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

    // Rebuild settings with correct error model for this experiment
    let poly = match experiment {
        1..=4 => ErrorPoly::new(0.0, 0.1, 0.0, 0.0),
        5 => ErrorPoly::new(0.0, 0.01, 0.0, 0.0),
        _ => panic!("Invalid experiment"),
    };
    let em = ErrorModel::proportional(poly, 1.0, None);
    let new_ems = ErrorModels::new().add(0, em).unwrap();

    let mut new_settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(settings.parameters().clone())
        .set_error_models(new_ems)
        .build();
    new_settings.set_cycles(usize::MAX);
    new_settings.set_cache(true);
    new_settings.set_prior(settings.prior().clone());
    new_settings.set_output_path(&format!(
        "examples/sde_paper/output/experiment{}",
        experiment
    ));
    new_settings.set_write_logs(true);
    new_settings.set_log_level(LogLevel::DEBUG);
    new_settings.initialize_logs().unwrap();

    let mut problem = dispatch_algorithm(new_settings, eqn, data).unwrap();
    let result = problem.fit().unwrap();
    result.write_outputs().unwrap();
}

fn main() {
    generate_data();
    fit_experiment(0);
    fit_experiment(5);
    /*
    for i in 0..=2 { // 0..=5 runs all of them
        println!("Start experiment {}",i);
        fit_experiment(i);
        // cp ~wyamada/src/lapk/PMcore/examples/sde_paper/output/data/experiment___.csv
        //    ~wyamada/src/lapk/PMcore/examples/sde_paper/output/output/experiment___/
    }
    */
}
