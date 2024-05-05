use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use lazy_static::lazy_static;
use dashmap::DashMap;
use crate::simulator::likelihood::SubjectPredictions;


const CACHE_SIZE: usize = 10000;

#[derive(Clone, Debug, PartialEq, Hash)]
struct CacheKey {
    subject: String,
    support_point: SupportPointHash,
}

impl Eq for CacheKey {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SupportPointHash(u64);

impl SupportPointHash {
    fn new(support_point: &Vec<f64>) -> Self {
        let mut hasher = DefaultHasher::new();
        // Hash each element in the Vec
        for value in support_point {
            value.to_bits().hash(&mut hasher);
        }
        // Get the resulting hash
        SupportPointHash(hasher.finish())
    }
}

lazy_static! {
    static ref CACHE: DashMap<CacheKey, SubjectPredictions> =
        DashMap::with_capacity(CACHE_SIZE);
}

pub fn get_entry(subject: &String, support_point: &Vec<f64>) -> Option<SubjectPredictions> {
    let cache_key = CacheKey {
        subject: subject.clone(),
        support_point: SupportPointHash::new(support_point),
    };

    // Check if the key already exists
    match CACHE.get(&cache_key) {
        Some(existing_entry) => Some(existing_entry.clone()),
        None => None,
    }
}

pub fn insert_entry(subject: &String, support_point: &Vec<f64>, predictions: SubjectPredictions) {
    let cache_key = CacheKey {
        subject: subject.clone(),
        support_point: SupportPointHash::new(support_point),
    };

    // Insert the new entry
    CACHE.insert(cache_key, predictions);
}