use csv::WriterBuilder;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
use std::fs::File;

// Cycles
pub struct CycleWriter {
    writer: csv::Writer<File>,
}

impl CycleWriter {
    pub fn new(file_path: &str, parameter_names: Vec<String>) -> CycleWriter {
        let file = File::create(file_path).unwrap();
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

        // Write headers
        writer.write_field("cycle").unwrap();
        writer.write_field("neg2ll").unwrap();
        writer.write_field("gamlam").unwrap();
        writer.write_field("nspp").unwrap();

        for param_name in &parameter_names {
            writer.write_field(format!("{}.mean", param_name)).unwrap();
            writer
                .write_field(format!("{}.median", param_name))
                .unwrap();
            writer.write_field(format!("{}.sd", param_name)).unwrap();
        }

        writer.write_record(None::<&[u8]>).unwrap();

        CycleWriter { writer }
    }

    pub fn write(&mut self, cycle: usize, objf: f64, gamma: f64, theta: &Array2<f64>) {
        self.writer.write_field(format!("{}", cycle)).unwrap();
        self.writer.write_field(format!("{}", -2. * objf)).unwrap();
        self.writer.write_field(format!("{}", gamma)).unwrap();
        self.writer
            .write_field(format!("{}", theta.nrows()))
            .unwrap();

        for param in theta.axis_iter(Axis(1)) {
            self.writer
                .write_field(format!("{}", param.mean().unwrap()))
                .unwrap();
        }

        for param in theta.axis_iter(Axis(1)) {
            self.writer
                .write_field(format!("{}", median(param.to_owned().to_vec())))
                .unwrap();
        }

        for param in theta.axis_iter(Axis(1)) {
            self.writer
                .write_field(format!("{}", param.std(1.)))
                .unwrap();
        }

        self.writer.write_record(None::<&[u8]>).unwrap();
    }

    pub fn flush(&mut self) {
        self.writer.flush().unwrap();
    }
}

pub fn posterior(psi: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let py = psi.dot(w);
    let mut post: Array2<f64> = Array2::zeros((psi.nrows(), psi.ncols()));
    post.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let elem = psi.get((i, j)).unwrap() * w.get(j).unwrap() / py.get(i).unwrap();
                    element.fill(elem);
                });
        });
    post
}

pub fn median(data: Vec<f64>) -> f64 {
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

pub fn population_mean_median(theta: &Array2<f64>, w: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut mean = Array1::zeros(theta.ncols());
    let mut median = Array1::zeros(theta.ncols());

    for (i, (mn, mdn)) in mean.iter_mut().zip(&mut median).enumerate() {
        // Calculate the weighted mean
        let col = theta.column(i).to_owned() * w.to_owned();
        *mn = col.sum();

        // Calculate the median
        let ct = theta.column(i);
        let mut tup: Vec<(f64, f64)> = Vec::new();
        for (ti, wi) in ct.iter().zip(w) {
            tup.push((*ti, *wi));
        }

        tup.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

        let mut wacc: Vec<f64> = Vec::new();
        let mut widx: usize = 0;

        for (i, (_, wi)) in tup.iter().enumerate() {
            let acc = wi + wacc.last().unwrap_or(&0.0);
            wacc.push(acc);

            if acc > 0.5 {
                widx = i;
                break;
            }
        }

        let acc2 = wacc.pop().unwrap();
        let acc1 = wacc.pop().unwrap();
        let par2 = tup.get(widx).unwrap().0;
        let par1 = tup.get(widx - 1).unwrap().0;
        let slope = (par2 - par1) / (acc2 - acc1);

        *mdn = par1 + slope * (0.5 - acc1);
    }

    (mean, median)
}

pub fn posterior_mean_median(
    theta: &Array2<f64>,
    psi: &Array2<f64>,
    w: &Array1<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let mut mean = Array2::zeros((0, theta.ncols()));
    let mut median = Array2::zeros((0, theta.ncols()));

    // Normalize psi to get probabilities of each spp for each id
    let mut psi_norm: Array2<f64> = Array2::zeros((0, psi.ncols()));
    for row in psi.axis_iter(Axis(0)) {
        let row_w = row.to_owned() * w.to_owned();
        let row_sum = row_w.sum();
        let row_norm = &row_w / row_sum;
        psi_norm.push_row(row_norm.view()).unwrap();
    }

    // Transpose normalized psi to get ID (col) by prob (row)
    let psi_norm_transposed = psi_norm.t();

    // For each subject..
    for probs in psi_norm_transposed.axis_iter(Axis(1)) {
        let mut post_mean: Vec<f64> = Vec::new();
        let mut post_median: Vec<f64> = Vec::new();

        // For each parameter
        for pars in theta.axis_iter(Axis(1)) {
            // Calculate the mean
            let weighted_par = &probs * &pars;
            let the_mean = weighted_par.sum();
            post_mean.push(the_mean);

            // Calculate the median
            let mut tup: Vec<(f64, f64)> = Vec::new();

            for (ti, wi) in pars.iter().zip(probs) {
                tup.push((*ti, *wi));
            }

            tup.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

            if tup.first().unwrap().1 >= 0.5 {
                tup.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
            }

            let mut wacc: Vec<f64> = Vec::new();
            let mut widx: usize = 0;

            for (i, (_, wi)) in tup.iter().enumerate() {
                let acc = wi + wacc.last().unwrap_or(&0.0);
                wacc.push(acc);

                if acc > 0.5 {
                    widx = i;
                    break;
                }
            }

            let acc2 = wacc.pop().unwrap();
            let acc1 = wacc.pop().unwrap();
            let par2 = tup.get(widx).unwrap().0;
            let par1 = tup.get(widx - 1).unwrap().0;
            let slope = (par2 - par1) / (acc2 - acc1);
            let the_median = par1 + slope * (0.5 - acc1);
            post_median.push(the_median);
        }

        mean.push_row(Array::from(post_mean.clone()).view())
            .unwrap();
        median
            .push_row(Array::from(post_median.clone()).view())
            .unwrap();
    }

    (mean, median)
}
