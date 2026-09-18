use std::{fs, io::Cursor};
use std::sync::{Mutex, LazyLock};

use byteorder::{BigEndian, ReadBytesExt};
use faer::rand::distr::uniform::{SampleRange, SampleUniform};
use statrs::distribution::Normal;


static FIXED_RNG_FILE: LazyLock<Mutex<String>> = LazyLock::new(|| {
    Mutex::new(String::new())
});

/// Assumes that FIXED_RNG_FILE contains bytes representing f64 stored in big endian
static SHARED_CURSOR: LazyLock<Mutex<Cursor<Vec<u8>>>> = LazyLock::new(|| {
    let path = FIXED_RNG_FILE.lock().unwrap();
    let bytes = fs::read(*path)
        .expect("Failed to read file");
    Mutex::new(Cursor::new(bytes))
});

/// Returns a value from 0 to 1, reading from SHARED_CURSOR
pub fn random() -> f64 {
    let cursor = SHARED_CURSOR.lock().unwrap();
    cursor.read_f64::<BigEndian>().unwrap()
}

/// sample value from standard normal distribution
pub fn standard_normal() -> f64 {
    Normal::standard().inverse_cdf(random())
}

/// sample value from chi-squared distribution
pub fn chi_squared() -> f64 {
    todo!("working on it")
}

/// sample from a random range
pub fn random_range<T, R>(range: R) -> T 
where
    T: SampleUniform,
    R: SampleRange<T>
{
    todo!("working on it")
}
