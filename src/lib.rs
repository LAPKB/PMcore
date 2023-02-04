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
            [0.26215363, 0.5368943, 0.7156023],
            [0.9713137, 0.4605986, 0.07547736],
            [0.6825801, 0.7782303, 0.8197627],
            [0.02261722, 0.101124406, 0.45159543],
            [0.21788204, 0.97947216, 0.1410861]
        ])
    }
}
