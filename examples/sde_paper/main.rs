// use std::fs::File;
// use std::path::Path;

// use pharmsol::equation;
// use plotly::common::Marker;
// use plotly::{layout, Plot, Scatter};
// use pmcore::prelude::*;
// use rand::rngs::StdRng;
// use rand::SeedableRng;
// use rand_distr::weighted::WeightedIndex;
// use rand_distr::{Distribution, Normal};

// fn model() -> equation::SDE {
//     equation::SDE::new(
//         |x, p, _t, dx, _rateiv, _cov| {
//             // automatically defined
//             fetch_params!(p, ke0, v0, _s1, _s2);
//             // mean reversion to ke0
//             dx[1] = -x[1] + ke0;
//             let ke = x[1];

//             // mean reversion to V
//             dx[2] = -x[2] + v0;
//             // let v = x[2];
//             // user defined
//             dx[0] = -ke * x[0];
//         },
//         |p, d| {
//             fetch_params!(p, _ke0, _v0, s1, s2);
//             d[1] = s1;
//             d[2] = s2;
//         },
//         |_p| lag! {},
//         |_p| fa! {},
//         |p, _t, _cov, x| {
//             fetch_params!(p, k0, v, _s1, _s2);
//             x[1] = k0;
//             x[2] = v;
//         },
//         |x, p, _t, _cov, y| {
//             fetch_params!(p, _k0, _v0, _s1, _s2);
//             let v = x[2];
//             y[0] = x[0] / v;
//         },
//         (3, 1),
//         1,
//     )
// }

// fn subject() -> Subject {
//     data::Subject::builder("id1")
//         .bolus(0.0, 20.0, 0)
//         .observation(0.0, -1.0, 0)
//         .repeat(5, 0.2)
//         .build()
// }
// fn sample_k0(rng: &mut StdRng, n1: Normal<f64>, n2: Normal<f64>) -> f64 {
//     let weights = [0.5, 0.5];
//     let dist = WeightedIndex::new(&weights).unwrap();

//     let component = dist.sample(rng);
//     match component {
//         0 => n1.sample(rng),
//         1 => n2.sample(rng),
//         _ => panic!("Invalid component"),
//     }
// }

// fn sample_v(rng: &mut StdRng, n3: Normal<f64>) -> f64 {
//     1.0 + 0.2 * n3.sample(rng)
// }

// fn write_samples_to_file(
//     file_name: &str,
//     k0_pop: Vec<f64>,
//     v_pop: Vec<f64>,
//     s1: f64,
//     s2: f64,
//     fixed_v: Option<f64>,
// ) {
//     let sde = model();
//     let subject = subject();
//     let mut data = Vec::new();
//     k0_pop
//         .iter()
//         .enumerate()
//         .zip(v_pop.iter())
//         .for_each(|((i, k0), v)| {
//             let spp = match fixed_v {
//                 Some(fixed_v) => vec![*k0, fixed_v, s1, s2],
//                 None => vec![*k0, *v, s1, s2],
//             };
//             let trajectories = sde.estimate_predictions(&subject, &spp);
//             let trajectory = trajectories.row(0);
//             let mut sb = data::Subject::builder(format!("id{}", i)).bolus(0.0, 20.0, 0);
//             for (t, point) in trajectory.iter().enumerate() {
//                 sb = sb.observation((t) as f64 * 0.2, point.prediction(), 0);
//             }
//             data.push(sb.build());
//         });
//     let data = data::Data::new(data);
//     data.write_pmetrics(&File::create(Path::new(file_name)).unwrap());
// }
// const N_SAMPLES: usize = 100;

// fn main() {
//     //k0 dist according to the paper F(K0) = 0.5*N(m_1,(s_1)^2) + 0.5*N(m_2,(s_2)^2)
//     let m1 = 0.5;
//     let s1 = 0.05;
//     let m2 = 1.5;
//     let s2 = 0.15;
//     let n1 = Normal::new(m1, s1).unwrap();
//     let n2 = Normal::new(m2, s2).unwrap();
//     let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//     let mut k0_pop = Vec::new();
//     let mut v_pop = Vec::new();

//     //V dist according to the paper F(V) = 1+0.2*N(0,1)
//     let n3 = Normal::new(0.0, 1.0).unwrap();

//     for _ in 0..N_SAMPLES {
//         k0_pop.push(sample_k0(&mut rng, n1, n2));
//         v_pop.push(sample_v(&mut rng, n3));
//     }

//     // plot
//     let trace = Scatter::new(k0_pop.clone(), v_pop.clone())
//         .mode(plotly::common::Mode::Markers)
//         .name("Population")
//         .marker(Marker::new().size(8).opacity(0.6));

//     let mut plot = Plot::new();
//     plot.add_trace(trace);
//     plot.set_layout(
//         layout::Layout::new()
//             .title("Population Distribution")
//             .x_axis(layout::Axis::new().title("k0"))
//             .y_axis(layout::Axis::new().title("V")),
//     );

//     plot.show();

//     // Now, let's create the data file.
//     //experiment 1 σ1 = 1, σ2 = 0, Vol ≡ 1
//     write_samples_to_file(
//         "examples/sde_paper/data/experiment1.csv",
//         k0_pop.clone(),
//         v_pop.clone(),
//         1.0,
//         0.0,
//         Some(1.0),
//     );
//     //experiment 2 σ1 = 1, σ2 = 0, Vol ∼ random
//     write_samples_to_file(
//         "examples/sde_paper/data/experiment2.csv",
//         k0_pop.clone(),
//         v_pop.clone(),
//         1.0,
//         0.0,
//         None,
//     );
//     //experiment 3 σ1 = 1, σ2 = 1/2, Vol ≡ 1
//     write_samples_to_file(
//         "examples/sde_paper/data/experiment3.csv",
//         k0_pop.clone(),
//         v_pop.clone(),
//         1.0,
//         0.5,
//         Some(1.0),
//     );
//     //experiment 4 σ1 = σ2 = 0, Vol ∼ random
//     write_samples_to_file(
//         "examples/sde_paper/data/experiment4.csv",
//         k0_pop.clone(),
//         v_pop.clone(),
//         0.0,
//         0.0,
//         None,
//     );
//     //experiment 5 σ1 = σ2 = 1/2, Vol ∼ random
//     write_samples_to_file(
//         "examples/sde_paper/data/experiment5.csv",
//         k0_pop.clone(),
//         v_pop.clone(),
//         0.5,
//         0.5,
//         None,
//     );
// }
