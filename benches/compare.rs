use pmcore::prelude::*;

use diol::prelude::*;
use settings::{Log, *};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(list![ode_tel, analytical_tel, ode_bke, analytical_bke], [1]);
    bench.run()?;
    Ok(())
}

pub fn analytical_bke(bencher: Bencher, len: usize) {
    let eq = equation::Analytical::new(
        one_compartment,
        |_p, _t, _cov| {},
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );
    let settings = bke_settings();
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    bencher.bench(|| {
        for _ in 0..len {
            let result = black_box(algorithm.fit().unwrap());
            assert!(result.cycles() == 96);
            assert!(result.objf() == -344.64028277953844);
        }
    });
}

pub fn ode_bke(bencher: Bencher, len: usize) {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );
    let settings = bke_settings();
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    bencher.bench(|| {
        for _ in 0..len {
            let result = black_box(algorithm.fit().unwrap());
            assert!(result.cycles() == 104);
            assert!(result.objf() == -348.69505647385495);
        }
    });
}

fn analytical_tel(bencher: Bencher, len: usize) {
    let eq = equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    let settings = tel_settings();
    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    bencher.bench(|| {
        for _ in 0..len {
            let result = black_box(algorithm.fit().unwrap());
            assert!(result.cycles() == 686);
            assert!(result.objf() == 432.95499351489167);
        }
    });
}

fn ode_tel(bencher: Bencher, len: usize) {
    let eq = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_cov!(cov, t,);
            fetch_params!(p, ka, ke, _tlag, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    let settings = tel_settings();
    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    bencher.bench(|| {
        for _ in 0..len {
            let result = black_box(algorithm.fit().unwrap());
            assert!(result.cycles() == 707);
            assert!(result.objf() == 432.9542531584738);
        }
    });
}

fn tel_settings() -> Settings {
    let settings = Settings {
        config: Config {
            cycles: 1000,
            algorithm: "NPAG".to_string(),
            cache: true,
            ..Default::default()
        },
        predictions: settings::Predictions::default(),
        log: Log {
            level: "warn".to_string(),
            file: "".to_string(),
            write: false,
        },
        prior: Prior {
            file: None,
            sampler: "sobol".to_string(),
            points: 2129,
            seed: 347,
        },
        output: Output {
            write: false,
            ..Default::default()
        },
        convergence: Default::default(),
        advanced: Default::default(),
        error: Error {
            value: 5.0,
            class: "proportional".to_string(),
            poly: (0.02, 0.05, -2e-04, 0.0),
        },
        parameters: {
            Parameters::new()
                .add("Ka".to_string(), 0.1, 0.3, false)
                .unwrap()
                .add("Ke".to_string(), 0.001, 0.1, false)
                .unwrap()
                .add("Tlag1".to_string(), 0.0, 4.00, false)
                .unwrap()
                .add("V".to_string(), 30.0, 120.0, false)
                .unwrap()
                .to_owned()
        },
    };
    settings.validate().unwrap();
    settings
}

fn bke_settings() -> Settings {
    let settings = Settings {
        config: Config {
            cycles: 1024,
            algorithm: "NPAG".to_string(),
            cache: true,
            include: None,
            exclude: None,
        },
        predictions: settings::Predictions::default(),
        log: Log {
            level: "warn".to_string(),
            file: "".to_string(),
            write: false,
        },
        prior: Prior {
            file: None,
            points: settings::Prior::default().points,
            sampler: "sobol".to_string(),
            ..Default::default()
        },
        output: Output {
            write: false,
            path: "output".to_string(),
        },
        convergence: Convergence::default(),
        advanced: Advanced::default(),
        error: Error {
            value: 0.0,
            class: "additive".to_string(),
            poly: (0.0, 0.05, 0.0, 0.0),
        },
        parameters: {
            Parameters::new()
                .add("Ke".to_string(), 0.001, 0.1, false)
                .unwrap()
                .add("V".to_string(), 25.0, 250.0, false)
                .unwrap()
                .to_owned()
        },
    };
    settings.validate().unwrap();
    settings
}
