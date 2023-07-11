use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rapl::{utils::random::NdarrRand, Ndarr};
use std::time::Duration;

fn matmul_benchmark(c: &mut Criterion) {

    let a_8 = black_box(NdarrRand::normal(0f64, 1f64, [4, 4], Some(1)));
    let b_8 = black_box(NdarrRand::normal(0f64, 1f64, [4, 4], Some(2)));

    let a_16 = black_box(NdarrRand::normal(0f64, 1f64, [16, 16], Some(1)));
    let b_16 = black_box(NdarrRand::normal(0f64, 1f64, [16, 16], Some(2)));

    let a_32 = black_box(NdarrRand::normal(0f64, 1f64, [32, 32], Some(1)));
    let b_32 = black_box(NdarrRand::normal(0f64, 1f64, [32, 32], Some(2)));

    let a_128 = black_box(NdarrRand::normal(0f64, 1f64, [128, 128], Some(1)));
    let b_128 = black_box(NdarrRand::normal(0f64, 1f64, [128, 128], Some(2)));


    let a_512 = black_box(NdarrRand::normal(0f64, 1f64, [512, 512], Some(1)));
    let b_512 = black_box(NdarrRand::normal(0f64, 1f64, [512, 512], Some(2)));

    let a_1024 = black_box(NdarrRand::normal(0f64, 1f64, [1024, 1014], Some(1)));
    let b_1024 = black_box(NdarrRand::normal(0f64, 1f64, [1024, 1014], Some(2)));
    c.bench_function("matmul f64-n8   ", |bench| {
        bench.iter(|| a_8.mat_mul(&b_8))
    });
    c.bench_function("matmul f64-n16  ", |bench| {
        bench.iter(|| a_16.mat_mul(&b_16))
    });
    c.bench_function("matmul f64-n32  ", |bench| {
        bench.iter(|| a_32.mat_mul(&b_32))
    });
    c.bench_function("matmul f64-n128 ", |bench| {
        bench.iter(|| a_128.mat_mul(&b_128))
    });
    c.bench_function("matmul f64-n512 ", |bench| {
        bench.iter(|| a_512.mat_mul(&b_512))
    });

    c.bench_function("matmul f64-n1024", |bench| {
        bench.iter(|| a_1024.mat_mul(&b_1024))
    });
}
criterion_group! {
  name = benches;
  config = Criterion::default().measurement_time(Duration::from_secs(7)).sample_size(10);
  targets = matmul_benchmark
}
criterion_main!(benches);
