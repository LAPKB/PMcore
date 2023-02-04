pub mod base;
// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

#[cfg(test)]
mod tests {
    use super::*;
    use base::*;
    use ndarray::array;

    #[test]
    fn basic_sobol(){
        assert_eq!(sobol_seq(5, 3, 347), array![
            [0.10731888, 0.14647412, 0.58510387],
            [0.9840305, 0.76333654, 0.19097507],
            [0.3847711, 0.73466134, 0.2616291],
            [0.70233, 0.41038263, 0.9158684],
            [0.60167587, 0.61712956, 0.62639713]
        ])
    }
}
