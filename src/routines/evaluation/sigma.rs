use ndarray::Array1;

/// Contains information on the observation error

/// ErrorModel contains the information on uncertainties in observations
///
/// The elements of the error polynomial corresponds to the terms in SD = C0 + C1 x obs + C2*obs^2 + C3*obs^3
///
/// See [ErrorType] for more information
#[derive(Debug, Clone)]
pub struct ErrorModel<'a> {
    pub c: (f64, f64, f64, f64),
    pub gl: f64,
    pub e_type: &'a ErrorType,
}

// Note to self, if we want to support multiple distribution samplers in the future
// the easiest way is to create a distribution enum and to put it into the ErrorModel.

/// ErrorType defines the current error model
///
/// # Multiplicative / Proportional
/// error = SD * γ (gamma)
///
/// # Additive
/// error = (SD<sup>2</sup> + lambda<sup>2</sup>)<sup>0.5</sup>
#[derive(Debug, Clone)]
pub enum ErrorType {
    Add,
    Prop,
}

//TODO: remove
/// Computes the error of an observation given its value, the error model, and the error polynomial
/// Observations are weighted by 1/error<sup>2</sup>
impl<'a> ErrorModel<'_> {
    pub fn sigma(&self, _yobs: &Array1<f64>) -> Array1<f64> {
        unimplemented!();
        // let alpha = self.c.0
        //     + self.c.1 * yobs
        //     + self.c.2 * yobs.mapv(|x| x.powi(2))
        //     + self.c.3 * yobs.mapv(|x| x.powi(3));

        // let res = match self.e_type {
        //     ErrorType::Add => (alpha.mapv(|x| x.powi(2)) + self.gl.powi(2)).mapv(|x| x.sqrt()),
        //     ErrorType::Prop => self.gl * alpha,
        // };

        // res.mapv(|x| {
        //     if x.is_nan() || x < 0.0 {
        //         tracing::error!(
        //             "The computed standard deviation is either NaN or negative (SD = {}), coercing to 0",
        //             x
        //         );
        //         0.0
        //     } else {
        //         x
        //     }
        // })
    }
}
