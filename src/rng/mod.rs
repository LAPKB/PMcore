use std::{fs, io::Cursor};
use std::sync::{Mutex, LazyLock};

use byteorder::{BigEndian, ReadBytesExt};
use statrs::distribution::{ContinuousCDF, Normal};


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
