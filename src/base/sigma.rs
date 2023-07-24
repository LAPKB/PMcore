use ndarray::Array1;

pub trait Sigma {
    fn sigma(&self, yobs: &Array1<f64>) -> Array1<f64>;
}

pub struct ErrorPoly<'a> {
    pub c: (f64, f64, f64, f64),
    pub gl: f64,
    pub e_type: &'a ErrorType,
}

pub enum ErrorType {
    Add,
    Prop,
}

impl<'a> Sigma for ErrorPoly<'a> {
    fn sigma(&self, yobs: &Array1<f64>) -> Array1<f64> {
        let alpha = self.c.0
            + self.c.1 * yobs
            + self.c.2 * yobs.mapv(|x| x.powi(2))
            + self.c.3 * yobs.mapv(|x| x.powi(3));

        let res = match self.e_type {
            ErrorType::Add => (alpha.mapv(|x| x.powi(2)) + self.gl.powi(2)).mapv(|x| x.sqrt()),
            ErrorType::Prop => self.gl * alpha,
        };

        res.mapv(|x| {
            if x.is_nan() || x < 0.0 {
                log::error!(
                    "The computed standard deviation is either NaN or negative!: {}",
                    x
                );
                x
            } else {
                x
            }
        })
    }
}
