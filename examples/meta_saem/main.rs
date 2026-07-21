//! Fit the parent/metabolite model with SAEM.
//!
//! PMcore              NONMEM                    Monolix
//! -------------------------------------------------------------------------
//! Parameter initial   Initial $THETA            Initial typical value
//! Parameter scale     ETA equation in $PK       Parameter distribution
//! Omega               $OMEGA                    Random-effect SD/correlation
//! Error model         $ERROR / $SIGMA           Observation error model
//!
//! `with_initial(...)`, `Omega::diagonal(...)`, and the error-model values
//! are initial estimates unless explicitly fixed.

use pmcore::prelude::*;

fn main() -> Result<()> {
    Logger::new().stdout(true).init()?;

    // Structural model: approximately NONMEM $DES or Monolix [LONGITUDINAL].
    let eq = ode! {
        name: "meta_saem",
        params: [cls, fm, k20, relv, theta1, theta2, vs],
        covariates: [wt, pkvisit],
        states: [central, metabolite],
        outputs: [outeq_1, outeq_2],
        routes: [
            infusion(input_1) -> central,
        ],
        diffeq: |x, _t, dx| {
            let cl =
                cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);

            let v =
                vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);

            let ke = cl / v;

            dx[central] = -ke * x[central] * (1.0 - fm) - fm * x[central];
            dx[metabolite] = fm * x[central] - k20 * x[metabolite];
        },
        out: |x, _t, y| {
            let cl =
                cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);

            let v =
                vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);

            let v2 = relv * v;
            let _ke = cl / v;

            y[outeq_1] = x[central] / v;
            y[outeq_2] = x[metabolite] / v2;
        },
    }
    .with_solver(OdeSolver::Bdf)
    .with_tolerances(1e-8, 1e-10);

    let data = data::read_pmetrics("examples/meta/meta.csv")?;

    // Parametric NLME model: estimates typical values, IIV, and residual error.
    let problem = EstimationProblem::parametric(eq, data)
        // Log-normal IIV:
        //   CLS_i = TVCLS * exp(ETA_CLS,i)
        // NONMEM: CLS = THETA(1) * EXP(ETA(1))
        // Monolix: distribution=logNormal, typical=TVCLS, sd=omega_CLS
        .parameter(Parameter::log("cls").with_initial(1.0))
        // Logit-normal IIV bounded to (0, 1):
        //   logit(FM_i) = logit(TVFM) + ETA_FM,i
        // NONMEM: explicit logit/inverse-logit transformation in $PK
        // Monolix: distribution=logitNormal, typical=TVFM, sd=omega_FM
        .parameter(Parameter::logit("fm", 0.0, 1.0).with_initial(0.20))
        // Log-normal IIV: K20_i = TVK20 * exp(ETA_K20,i)
        .parameter(Parameter::log("k20").with_initial(0.10))
        // Logit-normal IIV bounded to (0, 1).
        .parameter(Parameter::logit("relv", 0.0, 1.0).with_initial(0.50))
        // Fixed population coefficients without ETA.
        //
        // `pkvisit` is currently a covariate, not an occasion-level random
        // effect. Therefore, this model does not define IOV.
        //
        // NONMEM: fixed THETA values with no corresponding ETA
        // Monolix: fixed effects with no variability
        .parameter(
            Parameter::real("theta1")
                .with_initial(0.0)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::real("theta2")
                .with_initial(0.0)
                .fixed()
                .without_random_effect(),
        )
        // Log-normal IIV: VS_i = TVVS * exp(ETA_VS,i)
        .parameter(Parameter::log("vs").with_initial(2.0))
        // Initial IIV covariance matrix:
        //
        // NONMEM:
        //   $OMEGA DIAGONAL(5)
        //   0.10
        //   0.10
        //   0.10
        //   0.10
        //   0.10
        //
        // PMcore and NONMEM values are ETA variances.
        // Monolix uses ETA SDs, so 0.10 variance corresponds to:
        //   omega = sqrt(0.10) ≈ 0.316
        //
        // Undeclared covariances are fixed to zero.
        .omega(Omega::diagonal([
            ("cls", 0.10),
            ("fm", 0.10),
            ("k20", 0.10),
            ("relv", 0.10),
            ("vs", 0.10),
        ]))
        // Combined additive/proportional residual error.
        //
        // NONMEM: observation model in $ERROR with residual terms in $SIGMA
        // Monolix: combined observation error with initial a=0.50, b=0.10
        //
        // Each output has its own independently estimated error parameters.
        .error_model("outeq_1", ResidualErrorModel::combined(0.50, 0.10))
        .error_model("outeq_2", ResidualErrorModel::combined(0.50, 0.10))
        .build()?;

    // SAEM algorithm settings
    let config = SaemConfig::new()
        .seed(20_260_717)
        .n_chains(4)
        .mcmc_iterations(4)
        .eta_block_iterations(1)
        .burn_in(100)
        .k1_iterations(220)
        .k2_iterations(180)
        .averaged_iterates(0.75);

    let result = problem.fit_with(config)?;
    result.write_outputs("outputs/meta_saem", 0.0, 0.0)?;

    println!("termination: {:?}", result.termination_reason());
    println!("conditional N2LL: {:.6}", result.conditional_n2ll());

    // Final typical population values: NONMEM THETA / Monolix typical values.
    println!("population parameters:");
    for (name, value) in result
        .parameter_names()
        .iter()
        .zip(result.population_parameters())
    {
        println!("  {name}: {value:.6}");
    }

    // Final diagonal $OMEGA values, reported by PMcore as ETA variances.
    println!("IIV variances:");
    for (index, name) in result.random_effect_names().iter().enumerate() {
        println!("  {name}: {:.6}", result.omega()[[index, index]]);
    }

    // Final $SIGMA / Monolix observation-error parameter estimates.
    println!("residual models:");
    for estimate in result.residual_error_estimates() {
        println!("  {}: {:?}", estimate.output, estimate.model);
    }

    Ok(())
}
