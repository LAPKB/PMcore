use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SufficientStats {
	s1: Col<f64>,
	s2: Mat<f64>,
	s3: Col<f64>,
	stat_rese: f64,
	count: usize,
	n_obs: usize,
}

impl SufficientStats {
	pub fn new(n_params: usize) -> Self {
		Self {
			s1: Col::zeros(n_params),
			s2: Mat::zeros(n_params, n_params),
			s3: Col::zeros(n_params),
			stat_rese: 0.0,
			count: 0,
			n_obs: 0,
		}
	}

	pub fn reset(&mut self) {
		let n = self.s1.nrows();
		self.s1 = Col::zeros(n);
		self.s2 = Mat::zeros(n, n);
		self.s3 = Col::zeros(n);
		self.stat_rese = 0.0;
		self.count = 0;
		self.n_obs = 0;
	}

	pub fn npar(&self) -> usize {
		self.s1.nrows()
	}

	pub fn s1(&self) -> &Col<f64> {
		&self.s1
	}

	pub fn s1_mut(&mut self) -> &mut Col<f64> {
		&mut self.s1
	}

	pub fn s2(&self) -> &Mat<f64> {
		&self.s2
	}

	pub fn s2_mut(&mut self) -> &mut Mat<f64> {
		&mut self.s2
	}

	pub fn count(&self) -> usize {
		self.count
	}

	pub fn s3(&self) -> &Col<f64> {
		&self.s3
	}

	pub fn s3_mut(&mut self) -> &mut Col<f64> {
		&mut self.s3
	}

	pub fn stat_rese(&self) -> f64 {
		self.stat_rese
	}

	pub fn set_stat_rese(&mut self, value: f64) {
		self.stat_rese = value;
	}

	pub fn add_stat_rese(&mut self, value: f64) {
		self.stat_rese += value;
	}

	pub fn n_obs(&self) -> usize {
		self.n_obs
	}

	pub fn set_n_obs(&mut self, n: usize) {
		self.n_obs = n;
	}

	pub fn add_n_obs(&mut self, n: usize) {
		self.n_obs += n;
	}

	pub fn accumulate(&mut self, psi: &Col<f64>) -> Result<()> {
		let n = self.npar();

		if psi.nrows() != n {
			bail!(
				"Parameter vector length ({}) doesn't match statistics dimension ({})",
				psi.nrows(),
				n
			);
		}

		for i in 0..n {
			self.s1[i] += psi[i];
		}

		for i in 0..n {
			for j in 0..n {
				self.s2[(i, j)] += psi[i] * psi[j];
			}
		}

		for i in 0..n {
			self.s3[i] += psi[i] * psi[i];
		}

		self.count += 1;

		Ok(())
	}

	pub fn accumulate_batch(&mut self, samples: &[Col<f64>]) -> Result<()> {
		for sample in samples {
			self.accumulate(sample)?;
		}
		Ok(())
	}

	pub fn stochastic_update(&mut self, new_stats: &SufficientStats, step_size: f64) -> Result<()> {
		if self.npar() != new_stats.npar() {
			bail!(
				"Statistics dimension mismatch: {} vs {}",
				self.npar(),
				new_stats.npar()
			);
		}

		if step_size == 0.0 {
			return Ok(());
		}

		let n = self.npar();

		for i in 0..n {
			self.s1[i] += step_size * (new_stats.s1[i] - self.s1[i]);
		}

		for i in 0..n {
			for j in 0..n {
				self.s2[(i, j)] += step_size * (new_stats.s2[(i, j)] - self.s2[(i, j)]);
			}
		}

		for i in 0..n {
			self.s3[i] += step_size * (new_stats.s3[i] - self.s3[i]);
		}

		self.stat_rese += step_size * (new_stats.stat_rese - self.stat_rese);
		self.count = ((1.0 - step_size) * self.count as f64 + step_size * new_stats.count as f64)
			.round() as usize;
		self.n_obs = ((1.0 - step_size) * self.n_obs as f64 + step_size * new_stats.n_obs as f64)
			.round() as usize;

		Ok(())
	}

	pub fn compute_m_step(&self) -> Result<(Col<f64>, Mat<f64>)> {
		if self.count == 0 {
			bail!("Cannot compute M-step with zero samples");
		}

		let n = self.npar();
		let count_f64 = self.count as f64;
		let mu = Col::from_fn(n, |i| self.s1[i] / count_f64);
		let omega = Mat::from_fn(n, n, |i, j| self.s2[(i, j)] / count_f64 - mu[i] * mu[j]);

		Ok((mu, omega))
	}

	pub fn merge(&mut self, other: &SufficientStats) -> Result<()> {
		if self.npar() != other.npar() {
			bail!(
				"Cannot merge statistics with different dimensions: {} vs {}",
				self.npar(),
				other.npar()
			);
		}

		let n = self.npar();

		for i in 0..n {
			self.s1[i] += other.s1[i];
		}

		for i in 0..n {
			for j in 0..n {
				self.s2[(i, j)] += other.s2[(i, j)];
			}
		}

		for i in 0..n {
			self.s3[i] += other.s3[i];
		}

		self.stat_rese += other.stat_rese;
		self.count += other.count;
		self.n_obs += other.n_obs;

		Ok(())
	}
}

impl Default for SufficientStats {
	fn default() -> Self {
		Self::new(0)
	}
}

impl Serialize for SufficientStats {
	fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		use serde::ser::SerializeStruct;

		let mut state = serializer.serialize_struct("SufficientStats", 6)?;
		let s1_vec: Vec<f64> = (0..self.s1.nrows()).map(|i| self.s1[i]).collect();
		state.serialize_field("s1", &s1_vec)?;

		let s2_vec: Vec<Vec<f64>> = (0..self.s2.nrows())
			.map(|i| (0..self.s2.ncols()).map(|j| self.s2[(i, j)]).collect())
			.collect();
		state.serialize_field("s2", &s2_vec)?;

		let s3_vec: Vec<f64> = (0..self.s3.nrows()).map(|i| self.s3[i]).collect();
		state.serialize_field("s3", &s3_vec)?;
		state.serialize_field("stat_rese", &self.stat_rese)?;
		state.serialize_field("count", &self.count)?;
		state.serialize_field("n_obs", &self.n_obs)?;

		state.end()
	}
}

impl<'de> Deserialize<'de> for SufficientStats {
	fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		#[derive(Deserialize)]
		struct SufficientStatsData {
			s1: Vec<f64>,
			s2: Vec<Vec<f64>>,
			#[serde(default)]
			s3: Option<Vec<f64>>,
			#[serde(default)]
			stat_rese: f64,
			count: usize,
			#[serde(default)]
			n_obs: usize,
		}

		let data = SufficientStatsData::deserialize(deserializer)?;
		let n = data.s1.len();
		let s1 = Col::from_fn(n, |i| data.s1[i]);

		if data.s2.len() != n {
			return Err(serde::de::Error::custom("S2 row count doesn't match S1 length"));
		}

		let s2 = Mat::from_fn(n, n, |i, j| {
			if j < data.s2[i].len() {
				data.s2[i][j]
			} else {
				0.0
			}
		});

		let s3 = match data.s3 {
			Some(s3_data) if s3_data.len() == n => Col::from_fn(n, |i| s3_data[i]),
			_ => Col::zeros(n),
		};

		Ok(SufficientStats {
			s1,
			s2,
			s3,
			stat_rese: data.stat_rese,
			count: data.count,
			n_obs: data.n_obs,
		})
	}
}

#[derive(Debug, Clone, Copy)]
pub enum StepSizeSchedule {
	Constant(f64),
	Harmonic,
	RobbinsMonro { a: f64, b: f64 },
	PolyakRuppert { start_averaging: usize },
}

impl StepSizeSchedule {
	pub fn new_saem(n_burn_in: usize, _n_stochastic: usize) -> Self {
		StepSizeSchedule::PolyakRuppert {
			start_averaging: n_burn_in,
		}
	}

	pub fn step_size(&self, k: usize) -> f64 {
		match self {
			StepSizeSchedule::Constant(gamma) => *gamma,
			StepSizeSchedule::Harmonic => 1.0 / k as f64,
			StepSizeSchedule::RobbinsMonro { a, b } => a / (k as f64 + b),
			StepSizeSchedule::PolyakRuppert { start_averaging } => {
				if k < *start_averaging {
					1.0
				} else {
					1.0 / (k - start_averaging + 1) as f64
				}
			}
		}
	}
}

impl Default for StepSizeSchedule {
	fn default() -> Self {
		StepSizeSchedule::PolyakRuppert {
			start_averaging: 100,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_sufficient_stats_accumulation() {
		let mut stats = SufficientStats::new(2);
		let sample1 = Col::from_fn(2, |i| if i == 0 { 1.0 } else { 2.0 });
		let sample2 = Col::from_fn(2, |i| if i == 0 { 3.0 } else { 4.0 });

		stats.accumulate(&sample1).unwrap();
		stats.accumulate(&sample2).unwrap();

		assert_eq!(stats.count(), 2);
		assert_eq!(stats.s1()[0], 4.0);
		assert_eq!(stats.s1()[1], 6.0);
	}

	#[test]
	fn test_m_step_computation() {
		let mut stats = SufficientStats::new(2);

		for i in 0..3 {
			let sample = Col::from_fn(2, |j| (2 * i + j + 1) as f64);
			stats.accumulate(&sample).unwrap();
		}

		let (mu, _omega) = stats.compute_m_step().unwrap();
		assert!((mu[0] - 3.0).abs() < 1e-10);
		assert!((mu[1] - 4.0).abs() < 1e-10);
	}
}