use criterion::criterion_main;

mod examples;
use examples::bke::*;
use examples::tel::*;

criterion_main! {
    bke_group,
    tel_group,
}
