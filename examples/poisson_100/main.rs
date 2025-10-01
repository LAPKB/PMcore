use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ec50,Kk,Kg,Kknat);
            let max_cfu = 10000.0;
            let Volume = 80.0;
            if x[0] <= max_cfu {
                let iresponse = 1.0 - 1.0 / ( 1.0 + (x[0]/0.2).powf(2.0) );
                let ddd = 1.0 + ((48.0 - x[0])/50.0).exp();
                let eee = (max_cfu - (max_cfu - 7500.0)/ddd)/max_cfu;
                let gfact = iresponse * eee * (max_cfu - x[0])/max_cfu;
                let growth_rate = gfact * Kg;
                let mut eff = 0.0;
                if x[0] > 0.0 {
                    let hill = 2.718282;
                    eff = 1.0 / ( 1.0 + (ec50*Volume/x[1]).powf(hill));
                }
                // else {
                //     eff = 0.0;
                // }
                let decay_rate = Kknat + eff*Kk;
                dx[0] = ( growth_rate - decay_rate ) * x[0];
            }
            else {
              // x[0] = max_cfu; // x is not mutable
              dx[0] = max_cfu - x[0]; // = 0.0;
            }
            // if ( X(2) .lt. maxCFU ) then
            //  IResponse = 1.0d0 - 1.0d0 / ( 1.0d0 + (X(2)/0.2)**2 )
            //  ddd = 1.0d0 + exp((48 - X(2))/50)
            //  Gfact = IResponse * ( (maxCFU - (maxCFU - 7500)/ddd)/maxCFU )
            //  Gfact = Gfact * (maxCFU - X(2))/maxCFU
            //  GrowthRate = Gfact * Kg
            //  if ( X(1) .gt. 0.0d0 ) then
            //    EFF = 1.0d0 / ( 1.0d0 + (EC50*Volume/X(1))**HILL )
            //  else
            //    EFF = 0.0d0
            //  end if
            //  DecayRate = Kknat + EFF*Kk
            //  XP(2) = ( GrowthRate - DecayRate ) * X(2)
            // else 
            //   X(2) = maxCFU
            //   XP(2) = 0.0d0
            // end if
            dx[1] = -0.04 * x[1] + rateiv[0]; // drug input is [0], drug output is [1]
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 1000.0;
        },
        |x, _p, _t, _cov, y| {
            // fetch_params!(p, _ke, _v);
            y[0] = x[0]; // this is CFU, no adjustments necessary
        },
        (2, 1),
    );

    let params = Parameters::new()
        .add("ec50", 1.762668, 2.138798, false)
        .add("Kk", 1.540277, 1.772984, false)
        .add("Kg", 2.227923, 2.901866, false)
        .add("Kknat", 0.9137731, 1.4066447, false);

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_model(ErrorModel::Proportional, 12.718282, (0.01, 0.0025, 0.0, 0.0))
        .build();

    settings.set_cycles(1000);
    settings.set_prior(Prior::sobol(8233, 22)); // prime number near 2^13
    settings.set_output_path("/Users/wyamada/Documents/CHLA/poisson_paper/data/RustTesting_100/rust_output/");
    settings.set_write_logs(true);

    settings.write()?;

    // settings.enable_logs(stdout: bool, )
    settings.initialize_logs()?;
    let data = data::read_pmetrics("/Users/wyamada/Documents/CHLA/poisson_paper/data/RustTesting_100/rust_input/d18.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit()?;
    result.write_outputs()?;

    Ok(())
}
