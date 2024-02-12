use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use pmcore::routines::expansion::adaptative_grid::adaptative_grid;

// Define a benchmark function
fn benchmark_adaptive_grid(c: &mut Criterion) {
    c.bench_function("adaptive_grid_small", |b| {
        let mut theta = Array2::from_elem((50, 5), 0.5); // Small grid example
        let eps = 0.1;
        let ranges = vec![(0.0, 1.0), (0.0, 100.0)];
        let min_dist = 0.01;
        
        b.iter(|| {
            let _ = adaptative_grid(
                black_box(&mut theta),
                black_box(eps),
                black_box(&ranges),
                black_box(min_dist)
            );
        });
    });

    c.bench_function("adaptive_grid_large", |b| {
        let mut theta = Array2::from_elem((1000, 10), 0.5); // Larger grid example
        let eps = 0.1;
        let ranges = vec![(0.0, 1.0), (0.0, 100.0)];
        let min_dist = 0.01;
        
        b.iter(|| {
            let _ = adaptative_grid(
                black_box(&mut theta),
                black_box(eps),
                black_box(&ranges),
                black_box(min_dist)
            );
        });
    });
}

criterion_group!(benches, benchmark_adaptive_grid);
criterion_main!(benches);
