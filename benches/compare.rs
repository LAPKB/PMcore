use pmcore::prelude::*;

use diol::prelude::*;
use settings::*;
use toml::Table;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(list![analytical_bke, ode_bke], [1]);
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
    bencher.bench(|| {
        for _ in 0..len {
            let result = black_box(fit(eq.clone(), data.clone(), settings.clone()).unwrap());
            assert!(result.cycles == 96);
            assert!(result.objf == -344.64028277953844);
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
    bencher.bench(|| {
        for _ in 0..len {
            let result = black_box(fit(eq.clone(), data.clone(), settings.clone()).unwrap());
            assert!(result.cycles == 104);
            assert!(result.objf == -348.69505647385495);
        }
    });
}

fn bke_settings() -> Settings {
    let settings = Settings {
        config: Config {
            cycles: 1024,
            algorithm: "NPAG".to_string(),
            tui: false,
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
            ..settings::Prior::default()
        },
        output: Output {
            write: false,
            path: "output".to_string(),
        },
        convergence: Convergence::default(),
        advanced: Advanced::default(),
        random: Random {
            parameters: Table::from(
                [
                    (
                        "Ke".to_string(),
                        toml::Value::Array(vec![
                            toml::Value::Float(0.001),
                            toml::Value::Float(3.0),
                        ]),
                    ),
                    (
                        "V".to_string(),
                        toml::Value::Array(vec![
                            toml::Value::Float(25.0),
                            toml::Value::Float(250.0),
                        ]),
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
        },
        fixed: None,
        constant: None,
        error: Error {
            value: 0.0,
            class: "additive".to_string(),
            poly: (0.0, 0.05, 0.0, 0.0),
        },
    };
    settings.validate().unwrap();
    settings
}
