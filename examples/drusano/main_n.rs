use std::process::exit;

use anyhow::Error;
use argmin::{
    core::{CostFunction, Executor, TerminationReason, TerminationStatus},
    solver::neldermead::NelderMead,
};
use logger::setup_log;
use pmcore::prelude::*;

fn main() {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_3, init_4, init_5, h1s, h2s,
                h1r1, h2r2
            );
            // Sec
            let e50_2r1 = e50_2s;
            let e50_1r2 = e50_1s;
            let h2r1 = h2s;
            let h1r2 = h1s;
            let mut xm0best = 0.0;

            dx[0] = rateiv[0] - cl1 * x[0] / v1;
            dx[1] = rateiv[1] - cl2 * x[1] / v2;

            let xns = x[2];
            let xnr1 = x[3];
            let xnr2 = x[4];
            let e = 1.0 - (xns + xnr1 + xnr2) / popmax;
            let mut d1 = x[0] / v1;
            let mut d2 = x[1] / v2;
            let mut u = d1 / e50_1s;
            let mut v = d2 / e50_2s;
            let mut w = alpha_s * d1 * d2 / (e50_1s * e50_2s);
            let mut h1 = 1.0_f64 / h1s;
            let mut h2 = 1.0_f64 / h2s;
            let mut xx = (h1 + h2) / 2.0;
            if u < 1.0E-5 && v < 1.0E-5 {
                xm0best = 0.0;
            } else {
                if v < 0.0 {
                    xm0best = u.powf(1.0 / h1);
                }
                if u < 0.0 {
                    xm0best = v.powf(1.0 / h2);
                }

                if v > 0.0 && u > 0.0 {
                    let start = 0.00001;
                    let tol = 1.0e-10;
                    let step = -2.0 * start;
                    // CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                    let bm0 = BESTM0 {
                        u,
                        v,
                        w,
                        h1,
                        h2,
                        xx,
                    };
                    let (xm0best1, valmin1, iconv) = bm0.get_best(start, step);
                    if iconv == false {
                        // Output a message indicating no convergence on the selection of best M0 for s
                        println!(" NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.");

                        // Output a message indicating the XP(3) EQ...
                        println!(" FOR THE XP(3) EQ.... ");

                        // Output the values of XM0BEST1 and VALMIN1 with formatting
                        println!(" THE EST. FOR M0 FROM ELDERY WAS  {:>20.12}", xm0best1);
                        println!(" AND THIS GAVE A VALMIN OF {:>20.12}", valmin1);

                        // Output the values of D1, D2, U, V, W, ALPHA_S, H1, and H2 with formatting
                        println!(" NOTE THAT D1,D2 = {:>20.12} {:>20.12}", d1, d2);
                        println!(" U,V = {:>20.12} {:>20.12}", u, v);
                        println!(" W,ALPHA_S = {:>20.12} {:>20.12}", w, alpha_s);
                        println!(" H1,H2 = {:>20.12} {:>20.12}", h1, h2);

                        exit(-1);
                    }
                    if valmin1 < 1.0e-10 {
                        xm0best = xm0best1;
                    } else {
                        // CALL FINDM0(U,V,alpha_s,H1,H2,XM0EST)
                        let xm0est = find_m0(u, v, alpha_s, h1, h2);
                        if xm0est < 0.0 {
                            xm0best = xm0best1;
                        } else {
                            // START(1) = XM0EST
                            // STEP(1)= -.2D0*START(1)
                            // CALL ELDERY(1,START,XM0BEST2,VALMIN2,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                            let bm0 = BESTM0 {
                                u,
                                v,
                                w,
                                h1,
                                h2,
                                xx,
                            };
                            let (xm0best2, valmin2, iconv) = bm0.get_best(xm0est, -2.0 * xm0est);
                            xm0best = xm0best1;
                            if valmin2 < valmin1 {
                                xm0best = xm0best2;
                            }
                            if iconv == false {
                                panic!("NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.");
                            } //235
                        } //237
                    } //240
                } //243
            }
            let xms = xm0best / (xm0best + 1.0);
            dx[2] = xns * (kgs * e - kks * xms);

            d1 = x[0] / v1;
            d2 = x[1] / v2;
            u = d1 / e50_1r1;
            v = d2 / e50_2r1;
            w = alpha_r1 * d1 * d2 / (e50_1r1 * e50_2r1);
            h1 = 1.0_f64 / h1r1;
            h2 = 1.0_f64 / h2r1;
            xx = (h1 + h2) / 2.0;
            if u < 1.0e-5 && v < 1.0e-5 {
                xm0best = 0.0;
            } else {
                if v < 0.0 {
                    xm0best = u.powf(1.0 / h1);
                }
                if u < 0.0 {
                    xm0best = v.powf(1.0 / h2);
                }
                if v > 0.0 && u > 0.0 {
                    //START(1) = .00001
                    let tol = 1.0e-10;
                    // STEP(1)= -.2D0*START(1)
                    // CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                    let bm0 = BESTM0 {
                        u,
                        v,
                        w,
                        h1,
                        h2,
                        xx,
                    };
                    let (xm0best1, valmin1, iconv) = bm0.get_best(0.00001, -2.0 * 0.00001);
                    if iconv == false {
                        panic!("NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.");
                    }
                    if valmin1 < 1.0e-10 {
                        xm0best = xm0best1;
                    } else {
                        // CALL FINDM0(U,V,alpha_r1,H1,H2,XM0EST)
                        let xm0est = find_m0(u, v, alpha_s, h1, h2);
                        if xm0est < 0.0 {
                            xm0best = xm0best1;
                        } else {
                            // START(1) = XM0EST
                            // STEP(1)= -.2D0*START(1)
                            // CALL ELDERY(1,START,XM0BEST2,VALMIN2,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                            let bm0 = BESTM0 {
                                u,
                                v,
                                w,
                                h1,
                                h2,
                                xx,
                            };
                            let (xm0best2, valmin2, iconv) = bm0.get_best(xm0est, -2.0 * xm0est);
                            xm0best = xm0best1;
                            if valmin2 < valmin1 {
                                xm0best = xm0best2;
                            }
                            if iconv == false {
                                panic!("NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.");
                            } //235
                        } //237
                    } //240
                }
            }
            let xmr1 = xm0best / (xm0best + 1.0);
            dx[3] = xnr1 * (kgr1 * e - kkr1 * xmr1);

            d1 = x[0] / v1;
            d2 = x[1] / v2;
            u = d1 / e50_1r2;
            v = d2 / e50_2r2;
            w = alpha_r2 * d1 * d2 / (e50_1r2 * e50_2r2);
            h1 = 1.0_f64 / h1r2;
            h2 = 1.0_f64 / h2r2;
            xx = (h1 + h2) / 2.0;
            if u < 1.0e-5 && v < 1.0e-5 {
                xm0best = 0.0;
            } else {
                if v < 0.0 {
                    xm0best = u.powf(1.0 / h1);
                }
                if u < 0.0 {
                    xm0best = v.powf(1.0 / h2);
                }

                if v > 0.0 && u > 0.0 {
                    //START(1) = .00001
                    let tol = 1.0e-10;
                    // STEP(1)= -.2D0*START(1)
                    // CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                    let xm0best1 = 0.0;
                    let valmin1 = 0.0;
                    let iconv = 0.0;
                    if iconv == 0.0 {
                        panic!("NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.");
                    }
                    if valmin1 < 1.0e-10 {
                        xm0best = xm0best1;
                    } else {
                        // CALL FINDM0(U,V,alpha_s,H1,H2,XM0EST)
                        let xm0est = find_m0(u, v, alpha_s, h1, h2);
                        if xm0est < 0.0 {
                            xm0best = xm0best1;
                        } else {
                            // START(1) = XM0EST
                            // STEP(1)= -.2D0*START(1)
                            // CALL ELDERY(1,START,XM0BEST2,VALMIN2,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                            let xm0best2 = 0.0;
                            let valmin2 = 0.0;
                            let iconv = 0.0;
                            xm0best = xm0best1;
                            if valmin2 < valmin1 {
                                xm0best = xm0best2;
                            }
                            if iconv == 0.0 {
                                panic!("NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.");
                            } //235
                        } //237
                    } //240
                } //243
            }
            let xmr2 = xm0best / (xm0best + 1.0);
            dx[4] = xnr2 * (kgr2 * e - kkr2 * xmr2);
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, t, cov, x| {
            fetch_params!(
                p, v1, cl1, v2, cl2, popmax, kgs, kks, e50_1s, e50_2s, alpha_s, kgr1, kkr1,
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_3, init_4, init_5, h1s, h2s,
                h1r1, h2r2
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
                e50_1r1, alpha_r1, kgr2, kkr2, e50_2r2, alpha_r2, init_3, init_4, init_5, h1s, h2s,
                h1r1, h2r2
            );
            y[0] = x[0] / v1;
            y[1] = x[1] / v2;
            y[2] = (x[2] + x[3] + x[4]).log10();
            y[3] = x[3].log10();
            y[4] = x[4].log10();
        },
        (1, 1),
    );
    let settings = settings::read("examples/drusano/config.toml").unwrap();
    setup_log(&settings);
    let data = data::read_pmetrics("examples/drusano/drusano.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    let result = algorithm.fit().unwrap();
    result.write_outputs().unwrap();
}

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
        let other_point = start + step;
        let solver = NelderMead::new(vec![start, other_point])
            .with_sd_tolerance(0.0001)
            .unwrap();
        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(1000))
            // .add_observer(SlogLogger::term(), ObserverMode::Always)
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
