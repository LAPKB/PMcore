use anyhow::Error;
use argmin::{
    core::{CostFunction, Executor, TerminationReason, TerminationStatus},
    solver::neldermead::NelderMead,
};
use logger::setup_log;
use pmcore::prelude::*;
use std::process::exit;
#[allow(unused_variables)]
fn main() {
    let eq = equation::ODE::new(
        |x, p, t, dx, rateiv, _cov| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1,
                h2r2
            );

            let e50_2r1 = e50_2s;
            let e50_1r2 = e50_1s;
            let h2r1 = h2s;
            let h1r2 = h1s;

            dx[0] = rateiv[0] - cl1 * x[0] / v1;
            dx[1] = rateiv[1] - cl2 * x[1] / v2;

            let xns = x[2];
            let xnr1 = x[3];
            let xnr2 = x[4];
            let e = 1.0 - (xns + xnr1 + xnr2) / popmax;

            // Case s
            let u_s = x[0] / (v1 * e50_1s);
            let v_s = x[1] / (v2 * e50_2s);
            let w_s = alpha_s * u_s * v_s / (e50_1s * e50_2s);
            let xm0best = get_xm0best(u_s, v_s, w_s, 1.0 / h1s, 1.0 / h2s, alpha_s);
            dx[2] = xns * (kgs * e - kks * xm0best / (xm0best + 1.0));

            // Case r1
            let u_r1 = x[0] / (v1 * e50_1r1);
            let v_r1 = x[1] / (v2 * e50_2r1);
            let w_r1 = alpha_r1 * u_r1 * v_r1 / (e50_1r1 * e50_2r1);
            let xm0best = get_xm0best(u_r1, v_r1, w_r1, 1.0 / h1r1, 1.0 / h2r1, alpha_s);
            dx[3] = xnr1 * (kgr1 * e - kkr1 * xm0best / (xm0best + 1.0));

            // Case r2
            let u_r2 = x[0] / (v1 * e50_1r2);
            let v_r2 = x[1] / (v2 * e50_2r2);
            let w_r2 = alpha_r2 * u_r2 * v_r2 / (e50_1r2 * e50_2r2);
            let xm0best = get_xm0best(u_r2, v_r2, w_r2, 1.0 / h1r2, 1.0 / h2r2, alpha_s);
            dx[4] = xnr2 * (kgr2 * e - kkr2 * xm0best / (xm0best + 1.0));
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, t, cov, x| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1,
                h2r2
            );
            fetch_cov!(cov, t, ic_t);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 10.0_f64.powf(ic_t);
            x[3] = 10.0_f64.powf(init_4);
            x[4] = 10.0_f64.powf(init_5);
        },
        |x, p, _t, _cov, y| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_4, init_5, h1s, h2s, h1r1,
                h2r2
            );
            y[0] = x[0] / v1;
            y[1] = x[1] / v2;
            y[2] = (x[2] + x[3] + x[4]).log10();
            y[3] = x[3].log10();
            y[4] = x[4].log10();
        },
        (5, 5),
    );
    let settings = settings::read("examples/drusano/config.toml").unwrap();
    let _ = setup_log(&settings);
    let data = data::read_pmetrics("examples/drusano/data.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    algorithm.initialize().unwrap();
    while !match algorithm.next_cycle() {
        Ok(converged) => converged,
        Err((e, result)) => {
            eprintln!("{}", e);
            result.write_outputs().unwrap();
            panic!("Error during cycle");
        }
    } {}
    // while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}

#[derive(Debug, Clone)]
struct BESTM0 {
    u: f64,
    v: f64,
    w: f64,
    h1: f64,
    h2: f64,
    xx: f64,
}

impl CostFunction for BESTM0 {
    type Param = f64;
    type Output = f64;
    fn cost(&self, xm0: &Self::Param) -> Result<Self::Output, Error> {
        let t1 = self.u / xm0.powf(self.h1);
        let t2 = self.v / xm0.powf(self.h2);
        let t3 = self.w / xm0.powf(self.xx);
        Ok((1.0 - t1 - t2 - t3).powi(2))
    }
}

impl BESTM0 {
    fn get_best(self, start: f64, step: f64) -> (f64, f64, bool) {
        let solver = NelderMead::new(vec![start, start + step])
            .with_sd_tolerance(0.0001)
            .unwrap();
        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .unwrap();

        let converged = match res.state.termination_status {
            TerminationStatus::Terminated(reason) => match reason {
                TerminationReason::SolverConverged => true,
                _ => false,
            },
            _ => false,
        };

        (
            res.state.best_param.unwrap(),
            res.state.best_cost,
            converged,
        )
    }
}

fn find_m0(ufinal: f64, v: f64, alpha: f64, h1: f64, h2: f64) -> f64 {
    let noint = 1000;
    let delu = ufinal / (noint as f64);
    let mut xm = v.powf(1.0 / h2);
    let mut u = 0.0;
    let hh = (h1 + h2) / 2.0;

    for int in 1..=noint {
        let top = 1.0 / xm.powf(h1) + alpha * v / xm.powf(hh);
        let b1 = u * h1 / xm.powf(h1 + 1.0);
        let b2 = v * h2 / xm.powf(h2 + 1.0);
        let b3 = alpha * v * u * hh / xm.powf(hh + 1.0);
        let xmp = top / (b1 + b2 + b3);

        xm = xm + xmp * delu;

        if xm <= 0.0 {
            return -1.0; // Greco equation is not solvable
        }

        u = delu * (int as f64);
    }

    xm // Return the calculated xm0est
}

fn get_xm0best(u: f64, v: f64, w: f64, h1: f64, h2: f64, alpha_s: f64) -> f64 {
    let mut xm0best = 0.0;
    let xx = (h1 + h2) / 2.0;
    if u < 1.0e-5 && v < 1.0e-5 {
        return 0.0;
    } else {
        if v < 0.0 {
            xm0best = u.powf(1.0 / h1);
        }
        if u < 0.0 {
            xm0best = v.powf(1.0 / h2);
        }
        if v > 0.0 && u > 0.0 {
            let bm0 = BESTM0 {
                u,
                v,
                w,
                h1,
                h2,
                xx,
            };
            let (xm0best1, valmin1, iconv) = bm0.clone().get_best(0.00001, -2.0 * 0.00001);
            if !iconv {
                println!("NO CONVERGENCE ON SELECTION OF BEST M0.");
                println!("For THE XP(3) EQ .... ");
                println!("The Est. of M0 is {}", xm0best1);
                println!("The value of the function is {}", valmin1);
                println!("Note that U,V are {}, {}", u, v);

                exit(-1);
            }
            if valmin1 < 1.0e-10 {
                return xm0best1;
            } else {
                let xm0est = find_m0(u, v, alpha_s, h1, h2);
                if xm0est < 0.0 {
                    return xm0best1;
                } else {
                    let (xm0best2, valmin2, iconv) = bm0.get_best(xm0est, -2.0 * xm0est);
                    if !iconv {
                        panic!("NO CONVERGENCE ON SELECTION OF BEST M0.");
                    }
                    if valmin2 < valmin1 {
                        return xm0best2;
                    } else {
                        return xm0best1;
                    };
                }
            }
        }
    }
    xm0best
}
