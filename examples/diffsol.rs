use diffsol::*;
use nalgebra::DVector;
use std::rc::Rc;


struct FloatRange {
    current: f64,
    end: f64,
    step: f64,
}

impl FloatRange {
    fn new(start: f64, end: f64, step: f64) -> Self {
        FloatRange {
            current: start,
            end,
            step,
        }
    }
}

impl Iterator for FloatRange {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let next_val = self.current;
            self.current += self.step;
            Some(next_val)
        } else {
            None
        }
    }
}


pub fn main() {
    type T = f64;
    type V = DVector<T>;

    let ka = 1.0;
    let ke = 0.6;
    let p = V::from_vec(vec![ka, ke]);
    let mut problem = OdeSolverProblem::new_ode(
        // The rhs function `f`
        // In this case, a one-compartment model
        |x: &V, p: &V, _t: T, y: &mut V| {
            y[0] = -p[0] * x[0];
            y[1] = p[0] * x[0] - p[1] * x[1];
        },
        // The jacobian function `Jv`
        |x: &V, p: &V, _t: T, _v: &V, y: &mut V| {
            y[0] = -p[0] * x[0];
            y[1] = p[0] * x[0] - p[1] * x[1];
        },
        // The initial condition(s)
        |_p: &V, _t: T| V::from_vec(vec![100.0, 0.0]),
        p,
    );
    problem.rtol = 1.0e-4;
    problem.atol = Rc::new(V::from_vec(vec![1.0e-8, 1.0e-8]));

    let mut solver = Bdf::default();

    let mut state = OdeSolverState::new(&problem);
    solver.set_problem(&mut state, problem);


    let frange = FloatRange::new(0.0, 12.0, 0.5);

    let mut times: Vec<f64> = Vec::new();
    let mut preds: Vec<(f64,f64)> = Vec::new();
    for x in frange {
        let t = x;
        // Perform your operation with x
        while state.t <= t {
            solver.step(&mut state).unwrap();
        }
        let y = solver.interpolate(&state, t);
        times.push(t);
        preds.push((y[0], y[1]));
    }

    print!("Time, Y[0], Y[1]\n");
    for i in 0..times.len() {
        print!("{:.2}, {:.2}, {:.2}\n", times[i], preds[i].0, preds[i].1);
    }


}


