use std::process::exit;

use eyre::Result;
use np_core::prelude::{
    datafile::{Dose, Infusion},
    *,
};
use ode_solvers::*;

#[derive(Debug, Clone)]
struct Model<'a> {
    v1: f64,
    cl1: f64,
    v2: f64,
    cl2: f64,
    popmax: f64,
    kgs: f64,
    kks: f64,
    e50_1s: f64,
    e50_2s: f64,
    alpha_s: f64,
    kgr1: f64,
    kkr1: f64,
    e50_1r1: f64,
    alpha_r1: f64,
    kgr2: f64,
    kkr2: f64,
    e50_2r2: f64,
    alpha_r2: f64,
    init_3: f64,
    init_4: f64,
    init_5: f64,
    h1s: f64,
    h2s: f64,
    h1r1: f64,
    h2r2: f64,
    _scenario: &'a Scenario,
    infusions: Vec<Infusion>,
    dose: Option<Dose>,
}

type State = Vector5<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&mut self, t: Time, x: &mut State, dx: &mut State) {
        let mut rateiv = [0.0, 0.0];
        for infusion in &self.infusions {
            if t >= infusion.time && t <= (infusion.dur + infusion.time) {
                rateiv[infusion.compartment] = infusion.amount / infusion.dur;
            }
        }
        // Sec
        let e50_2r1 = self.e50_2s;
        let e50_1r2 = self.e50_1s;
        let h2r1 = self.h2s;
        let h1r2 = self.h1s;
        let mut xm0best = 0.0;

        ///////////////////// USER DEFINED ///////////////

        if x[0] < 0.0 {
            x[0] = 0.0;
        }
        if x[1] < 0.0 {
            x[1] = 0.0;
        }
        dx[0] = rateiv[0] - self.cl1 * x[0] / self.v1;
        dx[1] = rateiv[1] - self.cl2 * x[1] / self.v2;

        let xns = x[2];
        let xnr1 = x[3];
        let xnr2 = x[4];
        let e = 1.0 - (xns + xnr1 + xnr2) / self.popmax;
        let mut d1 = x[0] / self.v1;
        let mut d2 = x[1] / self.v2;
        let mut u = d1 / self.e50_1s;
        let mut v = d2 / self.e50_2s;
        let mut w = self.alpha_s * d1 * d2 / (self.e50_1s * self.e50_2s);
        let mut h1 = 1.0_f64 / self.h1s;
        let mut h2 = 1.0_f64 / self.h2s;
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
                //START(1) = .00001
                let tol = 1.0e-10;
                // STEP(1)= -.2D0*START(1)
                // CALL ELDERY(1,START,XM0BEST1,VALMIN1,TOL,STEP,1000,BESTM0,0,ICONV,NITER,ICNT)
                let xm0best1 = 0.0;
                let valmin1 = 0.0;
                let iconv = 0.0;
                if iconv == 0.0 {
                    // WRITE(*,9021)
                    // 9021 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.'/)

                    //         WRITE(*,*)' FOR THE XP(3) EQ.... '
                    //         WRITE(*,123) XM0BEST1,VALMIN1
                    // 123    FORMAT(/' THE EST. FOR M0 FROM ELDERY WAS  ',G20.12/
                    //     3' AND THIS GAVE A VALMIN OF ',G20.12//)
                    //         WRITE(*,129) D1,D2,U,V,W,ALPHA_S,H1,H2
                    // 129   FORMAT(//' NOTE THAT D1,D2 = ',2(G20.12,2X)/
                    //     1' U,V = ',2(G20.12,2X)/
                    //     2' W,ALPHA_S = ',2(G20.12,2X)/
                    //     3' H1,H2 = ',2(G20.12,2X))
                    // PAUSE
                    // STOP
                }
                if valmin1 < 1.0e-10 {
                    xm0best = xm0best1;
                } else {
                    // CALL FINDM0(U,V,alpha_s,H1,H2,XM0EST)
                    let xm0est = 0.0;
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
                            // WRITE(*,8021)
                            // 8021 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.'//
                            //     1' EVEN AFTER FINDM0 WAS USED. '/)
                            //       PAUSE
                            //       STOP
                        } //235
                    } //237
                } //240
            } //243
        }
        let xms = xm0best / (xm0best + 1.0);
        dx[2] = xns * (self.kgs * e - self.kks * xms);

        d1 = x[0] / self.v1;
        d2 = x[1] / self.v2;
        u = d1 / self.e50_1r1;
        v = d2 / e50_2r1;
        w = self.alpha_r1 * d1 * d2 / (self.e50_1r1 * e50_2r1);
        h1 = 1.0_f64 / self.h1r1;
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
                let xm0best1 = 0.0;
                let valmin1 = 0.0;
                let iconv = 0.0;
                if iconv == 0.0 {
                    // WRITE(*,9022)
                    // 9022 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.'/)
                    //     PAUSE
                    //     STOP
                }
                if valmin1 < 1.0e-10 {
                    xm0best = xm0best1;
                } else {
                    // CALL FINDM0(U,V,alpha_r1,H1,H2,XM0EST)
                    let xm0est = 0.0;
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
                            // WRITE(*,8022)
                            // 8022 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.'//
                            //     1' EVEN AFTER FINDM0 WAS USED. '/)
                            //       PAUSE
                            //       STOP
                        } //235
                    } //237
                } //240
            }
        }
        let xmr1 = xm0best / (xm0best + 1.0);
        dx[3] = xnr1 * (self.kgr1 * e - self.kkr1 * xmr1);

        d1 = x[0] / self.v1;
        d2 = x[1] / self.v2;
        u = d1 / e50_1r2;
        v = d2 / self.e50_2r2;
        w = self.alpha_r2 * d1 * d2 / (e50_1r2 * self.e50_2r2);
        h1 = 1.0_f64 / h1r2;
        h2 = 1.0_f64 / self.h2r2;
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
                    // WRITE(*,9022)
                    // 9022 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR r1.'/)
                    //     PAUSE
                    //     STOP
                }
                if valmin1 < 1.0e-10 {
                    xm0best = xm0best1;
                } else {
                    // CALL FINDM0(U,V,alpha_s,H1,H2,XM0EST)
                    let xm0est = 0.0;
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
                            // WRITE(*,8021)
                            // 8021 FORMAT(/' NO CONVERGENCE ON SELECTION OF BEST M0 FOR s.'//
                            //     1' EVEN AFTER FINDM0 WAS USED. '/)
                            //       PAUSE
                            //       STOP
                        } //235
                    } //237
                } //240
            } //243
        }
        let xmr2 = xm0best / (xm0best + 1.0);
        dx[4] = xnr2 * (self.kgr2 * e - self.kkr2 * xmr2);

        //////////////// END USER DEFINED ////////////////

        if let Some(dose) = &self.dose {
            if t >= dose.time {
                x[dose.compartment] += dose.amount;
                self.dose = None;
            }
        }
    }
}
#[derive(Debug, Clone)]
struct Sim {}

impl Simulate for Sim {
    fn simulate(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64> {
        let mut system = Model {
            v1: params[0],
            cl1: params[1],
            v2: params[2],
            cl2: params[3],
            popmax: params[4],
            kgs: params[5],
            kks: params[6],
            e50_1s: params[7],
            e50_2s: params[8],
            alpha_s: params[9],
            kgr1: params[10],
            kkr1: params[11],
            e50_1r1: params[12],
            alpha_r1: params[13],
            kgr2: params[14],
            kkr2: params[15],
            e50_2r2: params[16],
            alpha_r2: params[17],
            init_3: params[18],
            init_4: params[19],
            init_5: params[20],
            h1s: params[21],
            h2s: params[22],
            h1r1: params[23],
            h2r2: params[24],
            _scenario: scenario,
            infusions: vec![],
            dose: None,
        };
        let mut yout = vec![];
        let mut y0 = State::new(
            0.0,
            0.0,
            10.0_f64.powf(1.0),
            10.0_f64.powf(system.init_4),
            10.0_f64.powf(system.init_5),
        );
        let mut index: usize = 0;
        for block in &scenario.blocks {
            //if no code is needed here, remove the blocks from the codebase
            for event in &block.events {
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        system.infusions.push(Infusion {
                            time: event.time,
                            dur: event.dur.unwrap(),
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    } else {
                        //dose
                        system.dose = Some(Dose {
                            time: event.time,
                            amount: event.dose.unwrap(),
                            compartment: event.input.unwrap() - 1,
                        });
                    }
                } else if event.evid == 0 {
                    //obs
                    let v1 = params[0];
                    let v2 = params[2];
                    let out = match event.outeq.unwrap() {
                        1 => y0[0] / v1,
                        2 => y0[1] / v2,
                        3 => (y0[2] + y0[3] + y0[4]).log10(),
                        4 => y0[3].log10(),
                        5 => y0[4].log10(),
                        _ => {
                            log::error!("Invalid output equation");
                            exit(1)
                        }
                    };
                    yout.push(out);
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    let mut stepper = Rk4::new(system.clone(), event.time, y0, *next_time, 0.1);
                    let _res = stepper.integrate();
                    let y = stepper.y_out();
                    y0 = *y.last().unwrap();
                    index += 1;
                }
            }
        }
        yout
    }
}

fn main() -> Result<()> {
    // let scenarios = np_core::base::datafile::parse(&"examples/bimodal_ke.csv".to_string()).unwrap();
    // let scenario = scenarios.first().unwrap();
    start(
        Engine::new(Sim {}),
        "examples/bimodal_ke/config.toml".to_string(),
        (0.0, 0.05, 0.0, 0.0),
    )?;
    // let sim = Sim {};

    // // dbg!(&scenario);
    // dbg!(&scenario.obs);
    // dbg!(sim.simulate(vec![0.3142161965370178, 119.59214568138123], scenario));

    Ok(())
}
