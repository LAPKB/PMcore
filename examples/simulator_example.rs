// use plotly::{Plot, Scatter};
// use ode_solvers::*;
// use np_core::prelude::*;

// struct c1_pk<'a>{
//     ka: f64,
//     ke: f64,
//     scenario: &'a Scenario
// }

// type State = Vector2<f64>;
// type Time = f64;

// impl ode_solvers::System<State> for c1_pk<'_> {
//     fn system(&self, t: Time, X: &mut State, XP: &mut State) {

//         let ka = self.ka;
//         let ke = self.ke;
        
//         ///////////////////// USER DEFINED ///////////////

//         XP[0] = -ka*X[0];
//         XP[1] = ka*X[0] - ke*X[1];

//         //////////////// END USER DEFINED ////////////////
//         for index in 0..self.scenario.dose.len(){
//             if (t-self.scenario.time_dose[index] as f64).abs() < 1.0e-4 {
//                 X[0] = X[0]+self.scenario.dose[index] as f64;
//             }
//         }

//     }
// }

// struct Sim{
// }

// impl Simulate for Sim{
//     fn simulate(&self, params: Vec<f64>, y0: Vec<f64>, tspan:[f64;2], scenario: &Scenario) -> (Vec<f64>, Vec<f64>) {
//         dbg!(scenario);
//         let system = c1_pk {ka: params[0], ke: params[1], scenario: scenario};
        
    
//         let y0 = State::new(y0[0], y0[1]);
    
//         let mut stepper = Rk4::new(system, tspan[0], y0, tspan[1], 1.0e-3);
//         // let mut stepper = Dopri5::new(system, 0.0, 20.0, 1.0e-5, y0, 1.0e-14, 1.0e-14);
//         let _res = stepper.integrate();
//         // let stats = res.unwrap();

//         // println!("{}",stats);
//         let x = stepper.x_out().to_vec();
//         let y = stepper.y_out();
//         let yout: Vec<f64> = y.into_iter().map(|x| x.data.0[0][1] ).collect();
//         // // dbg!(yout);
//         // let mut plot = Plot::new();
//         // let trace = Scatter::new(x,yout);
//         // plot.add_trace(trace);

//         // plot.show();
  
//         (x, yout)    
//     }
// } 
fn main(){
    // let scenarios = datafile::parse("gendata.csv".to_string()).unwrap();
    // let scenario = scenarios.first().unwrap();
    // let engine = Engine::new(Sim{});
    // let _yout = engine.sim_obs(&scenario);

}