use std::io::{BufRead, Read};
use std::{fs, io::Cursor};
use std::sync::{Mutex, LazyLock};

static FIXED_RNG_FILE: LazyLock<Mutex<String>> = LazyLock::new(|| {
    Mutex::new(String::new())
});

static SHARED_CURSOR: LazyLock<Mutex<Cursor<String>>> = LazyLock::new(|| {
    let path = FIXED_RNG_FILE.lock().unwrap();
    let bytes = fs::read_to_string(*path)
        .expect("Failed to read file")
        // .split_whitespace()
        // .map(|s| s.parse::<u8>())
        // .collect()
        ;
    Mutex::new(Cursor::new(bytes))
});

fn test() {
    let cursor = SHARED_CURSOR.lock().unwrap();
    let mut input = String::new();
    (*cursor).read_line(&mut input).expect("failed to read message");
    let num: f64 = input.trim().parse().unwrap();
}
