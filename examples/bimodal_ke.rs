use ode_solvers::*;
use np_core::prelude::*;

struct Model<'a>{
    ke: f64,
    _v: f64,
    _scenario: &'a Scenario
}

type State = Vector1<f64>;
type Time = f64;

impl ode_solvers::System<State> for Model<'_> {
    fn system(&self, t: Time, y: &mut State, dy: &mut State) {
        let ke = self.ke;
        // let t = t - self.lag;
        
        let mut rateiv = [0.0, 0.0];
        rateiv[0] = if t>=0.0 && t<= 0.5 {500.0/0.5} else{0.0};
        // for index in 0..self.scenario.infusion.len(){
        //     if t >= self.scenario.time_infusion[index] &&
        //        t <= (self.scenario.time_infusion[index] - self.scenario.infusion[index].1) {
        //         rateiv[self.scenario.infusion[index].2] =
        //             self.scenario.infusion[index].0 / self.scenario.infusion[index].1;
        //     }
        // }

        ///////////////////// USER DEFINED ///////////////

        dy[0] = - ke*y[0] + rateiv[0];

        //////////////// END USER DEFINED ////////////////
        // for index in 0..self.scenario.dose.len(){
        //     if (t-self.scenario.time_dose[index] as f64).abs() < 1.0e-4 {
        //         y[self.scenario.dose[index].1] = y[self.scenario.dose[index].1]+self.scenario.dose[index].0 as f64;
        //     }
        // }
    }
}

struct Sim{}

impl Simulate for Sim{
    fn simulate(&self, params: Vec<f64>, tspan:[f64;2], scenario: &Scenario) -> (Vec<f64>, Vec<f64>) {
        let system = Model {ke: params[0], _v: params[1], _scenario: scenario};
        let y0 = State::new(0.0);
        let mut stepper = Rk4::new(system, tspan[0], y0, tspan[1],0.1);
        let _res = stepper.integrate();
        let x = stepper.x_out().to_vec();
        let y = stepper.y_out();
        let yout: Vec<f64> = y.into_iter().map(|y| {
            y[0]/params[1]
        } ).collect();
        (x, yout)    
    }
} 

fn main(){
    npag(Engine::new(Sim{}),
        vec![(0.001,50.0),(2.0,250.0)],
        "examples/bimodal_ke.toml".to_string(),
        347,
        (0.3,0.1,0.0,0.0)
    );
}
