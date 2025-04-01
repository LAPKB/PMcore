use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;

use pmcore::algorithms::npag::burke;
use pmcore::prelude::data::read_pmetrics;
use pmcore::prelude::*;
use pmcore::routines::initialization::sobol::generate;
use pmcore::routines::output::posterior;
use pmcore::structs::psi::calculate_psi;
use pmcore::structs::theta::Theta;

// Create a structure we can use to optimize
pub struct DoseOptim {
    dose: f64,
    target_time: f64,
    target_conc: f64,
    d_pop: Theta,
    d_flat: Theta,
    eq: ODE,
}

impl DoseOptim {
    pub fn new(
        dose: f64,
        target_time: f64,
        target_conc: f64,
        d_pop: Theta,
        d_flat: Theta,
        eq: ODE,
    ) -> Self {
        Self {
            dose,
            target_time,
            target_conc,
            d_pop,
            d_flat,
            eq,
        }
    }
}

impl CostFunction for DoseOptim {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        print!("Evaluating dose: {:?}", param);
        use pharmsol::ErrorModel;
        let predsub = Subject::builder("Johnny Bravo")
            .bolus(0.0, *param, 0)
            .observation(self.target_time, self.target_conc, 0)
            .build();

        let errmod = ErrorModel::new((0.01, 0.1, 0.0, 0.0), 1.0, &ErrorType::Add);

        let psi = calculate_psi(
            &self.eq,
            &Data::new(vec![predsub.clone()]),
            &self.d_pop,
            &errmod,
            false,
            true,
        );

        println!("Psi: {:#?}", psi);

        let (w, objf) = burke(&psi)?;

        println!("Objective function value: {:#?}", objf);
        println!("W: {:#?}", w);

        let posterior = posterior(&psi, &w);

        println!("Posterior: {:#?}", posterior);

        let mut toterr = 0.0;

        for spp in self.d_pop.matrix().row_iter() {
            let point: Vec<f64> = spp.iter().copied().collect();

            let pred = self.eq.simulate_subject(&predsub, &point, Some(&errmod)).0;

            toterr += pred.squared_error();
        }

        Ok(toterr)
    }
}

fn main() -> Result<()> {
    //Create a model with data
    let data = read_pmetrics("examples/theophylline/theophylline.csv")?;

    let eq = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ka, ke, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );

    let params = Parameters::new()
        .add("ka", 0.0, 3.0, false)
        .add("ke", 0.001, 3.0, false)
        .add("v", 0.0, 250.0, false);

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params.clone())
        .set_error_model(ErrorModel::Proportional, 5.0, (0.01, 0.1, 0.0, 0.0))
        .build();

    settings.set_cycles(1000);
    settings.set_prior_sampler(Sampler::Sobol, 2048, 22);

    let mut algorithm = dispatch_algorithm(settings, eq.clone(), data)?;

    println!("Fititng model...");

    let result = algorithm.fit()?;

    println!("Finished fititng model...");

    // Create D_pop
    let d_pop = result.get_theta().clone();
    println!("Theta: {:?}", d_pop);

    // Create D_flat
    let d_flat = generate(&params, 100, 22)?;

    // Create a dose optimizer
    let dose_optim = DoseOptim::new(4.0, 24.0, 3.0, d_pop, d_flat, eq);

    let solver = BrentOpt::new(0.0, 100.0);

    println!("Optimizing dose...");
    let res = Executor::new(dose_optim, solver)
        .configure(|state| state.max_iters(100))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    println!("#######################");
    println!("Finished optimizing dose...");

    println!("Optimal dose: {:?}", res.state.best_param);
    println!("{:#?}", res.state.best_cost);
    println!("{:#?}", res.state.counts);

    println!("Optimization finished");

    Ok(())
}
