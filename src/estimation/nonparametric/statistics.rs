use anyhow::{Result, bail};
use ndarray::{Array, Array1, Array2, Axis};

pub fn median(data: &[f64]) -> f64 {
    let mut data: Vec<f64> = data.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let size = data.len();
    match size {
        even if even % 2 == 0 => {
            let fst = data.get(even / 2 - 1).unwrap();
            let snd = data.get(even / 2).unwrap();
            (fst + snd) / 2.0
        }
        odd => *data.get(odd / 2_usize).unwrap(),
    }
}

pub fn weighted_median(data: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(
        data.len(),
        weights.len(),
        "The length of data and weights must be the same"
    );
    assert!(
        weights.iter().all(|&x| x >= 0.0),
        "Weights must be non-negative, weights: {:?}",
        weights
    );

    let mut weighted_data: Vec<(f64, f64)> = data
        .iter()
        .zip(weights.iter())
        .map(|(&d, &w)| (d, w))
        .collect();

    weighted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_weight: f64 = weights.iter().sum();
    let mut cumulative_sum = 0.0;

    for (i, &(_, weight)) in weighted_data.iter().enumerate() {
        cumulative_sum += weight;

        if cumulative_sum == total_weight / 2.0 {
            if i + 1 < weighted_data.len() {
                return (weighted_data[i].0 + weighted_data[i + 1].0) / 2.0;
            } else {
                return weighted_data[i].0;
            }
        } else if cumulative_sum > total_weight / 2.0 {
            return weighted_data[i].0;
        }
    }

    unreachable!("The function should have returned a value before reaching this point.");
}

pub fn population_mean_median(theta: &Array2<f64>, w: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
    let w = if w.is_empty() {
        tracing::warn!("w.len() == 0, setting all weights to 1/n");
        Array1::from_elem(theta.nrows(), 1.0 / theta.nrows() as f64)
    } else {
        w.clone()
    };

    if theta.nrows() != w.len() {
        bail!(
            "Number of parameters and number of weights do not match. Theta: {}, w: {}",
            theta.nrows(),
            w.len()
        );
    }

    let mut mean = Array1::zeros(theta.ncols());
    let mut median = Array1::zeros(theta.ncols());

    for (i, (mn, mdn)) in mean.iter_mut().zip(&mut median).enumerate() {
        let col = theta.column(i).to_owned() * w.to_owned();
        *mn = col.sum();

        let ct = theta.column(i);
        let mut params = vec![];
        let mut weights = vec![];
        for (ti, wi) in ct.iter().zip(w.clone()) {
            params.push(*ti);
            weights.push(wi);
        }

        *mdn = weighted_median(&params, &weights);
    }

    Ok((mean, median))
}

pub fn posterior_mean_median(
    theta: &Array2<f64>,
    psi: &Array2<f64>,
    w: &Array1<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let mut mean = Array2::zeros((0, theta.ncols()));
    let mut median = Array2::zeros((0, theta.ncols()));

    let w = if w.is_empty() {
        tracing::warn!("w is empty, setting all weights to 1/n");
        Array1::from_elem(theta.nrows(), 1.0 / theta.nrows() as f64)
    } else {
        w.clone()
    };

    if theta.nrows() != w.len() || theta.nrows() != psi.ncols() || psi.ncols() != w.len() {
        bail!("Number of parameters and number of weights do not match, theta.nrows(): {}, w.len(): {}, psi.ncols(): {}", theta.nrows(), w.len(), psi.ncols());
    }

    let mut psi_norm: Array2<f64> = Array2::zeros((0, psi.ncols()));
    for (i, row) in psi.axis_iter(Axis(0)).enumerate() {
        let row_w = row.to_owned() * w.to_owned();
        let row_sum = row_w.sum();
        let row_norm = if row_sum == 0.0 {
            tracing::warn!("Sum of row {} of psi is 0.0, setting that row to 1/n", i);
            Array1::from_elem(psi.ncols(), 1.0 / psi.ncols() as f64)
        } else {
            &row_w / row_sum
        };
        psi_norm.push_row(row_norm.view())?;
    }
    if psi_norm.iter().any(|&x| x.is_nan()) {
        dbg!(&psi);
        bail!("NaN values found in psi_norm");
    };

    for probs in psi_norm.axis_iter(Axis(0)) {
        let mut post_mean: Vec<f64> = Vec::new();
        let mut post_median: Vec<f64> = Vec::new();

        for pars in theta.axis_iter(Axis(1)) {
            let weighted_par = &probs * &pars;
            let the_mean = weighted_par.sum();
            post_mean.push(the_mean);

            let median = weighted_median(&pars.to_vec(), &probs.to_vec());
            post_median.push(median);
        }

        mean.push_row(Array::from(post_mean.clone()).view())?;
        median.push_row(Array::from(post_median.clone()).view())?;
    }

    Ok((mean, median))
}

#[cfg(test)]
mod tests {
    use super::{median, weighted_median};

    #[test]
    fn test_median_odd() {
        let data = vec![1.0, 3.0, 2.0];
        assert_eq!(median(&data), 2.0);
    }

    #[test]
    fn test_median_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data), 2.5);
    }

    #[test]
    fn test_median_single() {
        let data = vec![42.0];
        assert_eq!(median(&data), 42.0);
    }

    #[test]
    fn test_median_sorted() {
        let data = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        assert_eq!(median(&data), 15.0);
    }

    #[test]
    fn test_median_unsorted() {
        let data = vec![10.0, 30.0, 20.0, 50.0, 40.0];
        assert_eq!(median(&data), 30.0);
    }

    #[test]
    fn test_median_with_duplicates() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data), 2.0);
    }

    #[test]
    fn test_weighted_median_simple() {
        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![0.2, 0.5, 0.3];
        assert_eq!(weighted_median(&data, &weights), 2.0);
    }

    #[test]
    fn test_weighted_median_even_weights() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        assert_eq!(weighted_median(&data, &weights), 2.5);
    }

    #[test]
    fn test_weighted_median_single_element() {
        let data = vec![42.0];
        let weights = vec![1.0];
        assert_eq!(weighted_median(&data, &weights), 42.0);
    }

    #[test]
    #[should_panic(expected = "The length of data and weights must be the same")]
    fn test_weighted_median_mismatched_lengths() {
        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![0.1, 0.2];
        weighted_median(&data, &weights);
    }

    #[test]
    fn test_weighted_median_all_same_elements() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(weighted_median(&data, &weights), 5.0);
    }

    #[test]
    #[should_panic(expected = "Weights must be non-negative")]
    fn test_weighted_median_negative_weights() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.2, -0.5, 0.5, 0.8];
        assert_eq!(weighted_median(&data, &weights), 4.0);
    }
}