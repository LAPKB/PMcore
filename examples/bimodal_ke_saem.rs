//! Fit the `bimodal_ke` model and dataset with SAEM.
//!
//! SAEM assumes one Gaussian random-effects population, so this fit summarizes
//! the deliberately bimodal elimination-rate distribution.

use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    Logger::new().stdout(true).init()?;

    let eq = ode! {
        name: "bimodal_ke_saem",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [
            infusion(input_1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;

    let problem = EstimationProblem::parametric(eq, data)
        .parameter(Parameter::log("ke").with_initial(0.15))
        .parameter(Parameter::log("v").with_initial(120.0))
        .omega(Omega::diagonal([("ke", 0.5), ("v", 0.1)]))
        .error_model("outeq_1", ResidualErrorModel::proportional(0.15))
        .build()?;

    let config = SaemConfig::new()
        .seed(20_260_714)
        .n_chains(4)
        .mcmc_iterations(4)
        .burn_in(100)
        .k1_iterations(180)
        .k2_iterations(80);

    let result = problem.fit_with(config)?;
    result.write_outputs("outputs/bimodal_ke_saem", 0.0, 0.0)?;

    println!("population ke: {:.6}", result.population_parameters()[0]);
    println!("population v: {:.6}", result.population_parameters()[1]);
    println!("omega ke: {:.6}", result.omega()[[0, 0]]);
    println!("omega v: {:.6}", result.omega()[[1, 1]]);
    println!("proportional sigma: {:.6}", result.residual_sigmas()[0]);
    println!("conditional N2LL: {:.6}", result.conditional_n2ll());
    println!("termination: {:?}", result.termination_reason());

    Ok(())
}
