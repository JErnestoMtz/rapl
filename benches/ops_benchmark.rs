use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rapl::{utils::random::NdarrRand, Ndarr};
use std::time::Duration;

fn matmul_benchmark(c: &mut Criterion) {

    let fa_8 = black_box(NdarrRand::normal(0f64, 1f64, [4, 4], Some(1)));
    let fb_8 = black_box(NdarrRand::normal(0f64, 1f64, [4, 4], Some(2)));

    let fa_16 = black_box(NdarrRand::normal(0f64, 1f64, [16, 16], Some(1)));
    let fb_16 = black_box(NdarrRand::normal(0f64, 1f64, [16, 16], Some(2)));

    let fa_32 = black_box(NdarrRand::normal(0f64, 1f64, [32, 32], Some(1)));
    let fb_32 = black_box(NdarrRand::normal(0f64, 1f64, [32, 32], Some(2)));

    let fa_128 = black_box(NdarrRand::normal(0f64, 1f64, [128, 128], Some(1)));
    let fb_128 = black_box(NdarrRand::normal(0f64, 1f64, [128, 128], Some(2)));


    let fa_512 = black_box(NdarrRand::normal(0f64, 1f64, [512, 512], Some(1)));
    let fb_512 = black_box(NdarrRand::normal(0f64, 1f64, [512, 512], Some(2)));

    let fa_1024 = black_box(NdarrRand::normal(0f64, 1f64, [1024, 1024], Some(1)));
    let fb_1024 = black_box(NdarrRand::normal(0f64, 1f64, [1024, 1024], Some(2)));
    c.bench_function("matmul f64-n8   ", |bench| {
        bench.iter(|| fa_8.mat_mul(&fb_8))
    });
    c.bench_function("matmul f64-n16  ", |bench| {
        bench.iter(|| fa_16.mat_mul(&fb_16))
    });
    c.bench_function("matmul f64-n32  ", |bench| {
        bench.iter(|| fa_32.mat_mul(&fb_32))
    });
    c.bench_function("matmul f64-n128 ", |bench| {
        bench.iter(|| fa_128.mat_mul(&fb_128))
    });
    c.bench_function("matmul f64-n512 ", |bench| {
        bench.iter(|| fa_512.mat_mul(&fb_512))
    });

    c.bench_function("matmul f64-n1024", |bench| {
        bench.iter(|| fa_1024.mat_mul(&fb_1024))
    });

    let ia_8 = black_box(NdarrRand::choose(&[1,2,3,4,5], [4,4], Some(1)));
    let ib_8 = black_box(NdarrRand::choose(&[1,2,3,4,5], [4,4], Some(2)));

    let ia_16 = black_box(NdarrRand::choose(&[1,2,3,4,5], [16,16], Some(1)));
    let ib_16 = black_box(NdarrRand::choose(&[1,2,3,4,5], [16,16], Some(2)));

    let ia_32 = black_box(NdarrRand::choose(&[1,2,3,4,5], [32,32], Some(1)));
    let ib_32 = black_box(NdarrRand::choose(&[1,2,3,4,5], [32,32], Some(2)));

    let ia_64 = black_box(NdarrRand::choose(&[1,2,3,4,5], [64,64], Some(1)));
    let ib_64 = black_box(NdarrRand::choose(&[1,2,3,4,5], [64,64], Some(2)));
    
    let ia_128 = black_box(NdarrRand::choose(&[1,2,3,4,5], [128,128], Some(1)));
    let ib_128 = black_box(NdarrRand::choose(&[1,2,3,4,5], [128,128], Some(2)));

    let ia_512 = black_box(NdarrRand::choose(&[1,2,3,4,5], [512,512], Some(1)));
    let ib_512 = black_box(NdarrRand::choose(&[1,2,3,4,5], [512,512], Some(2)));

    let ia_1024 = black_box(NdarrRand::choose(&[1,2,3,4,5], [1024,1024], Some(1)));
    let ib_1024 = black_box(NdarrRand::choose(&[1,2,3,4,5], [1024,1024], Some(2)));
    c.bench_function("matmul i32-n8   ", |bench| {
        bench.iter(|| ia_8.mat_mul(&ib_8))});

    c.bench_function("matmul i32-n16   ", |bench| {
        bench.iter(|| ia_16.mat_mul(&ib_16))});

    c.bench_function("matmul i32-n32   ", |bench| {
        bench.iter(|| ia_32.mat_mul(&ib_32))});

    c.bench_function("matmul i64-n64   ", |bench| {
        bench.iter(|| ia_64.mat_mul(&ib_64))});
    
    c.bench_function("matmul i64-n128   ", |bench| {
        bench.iter(|| ia_128.mat_mul(&ib_128))});

    c.bench_function("matmul i64-n512   ", |bench| {
        bench.iter(|| ia_512.mat_mul(&ib_512))});

    c.bench_function("matmul i64-n1024   ", |bench| {
        bench.iter(|| ia_1024.mat_mul(&ib_1024))});
}
criterion_group! {
  name = benches;
  config = Criterion::default().measurement_time(Duration::from_secs(5)).sample_size(10);
  targets = matmul_benchmark
}
criterion_main!(benches);
