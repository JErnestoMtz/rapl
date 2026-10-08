//! conv2d + 2x2 max pooling as pure view algebra: `Win` windows, one permute, one fused contraction — O(output) memory, no convolution code.

use rapl::*;

/// x: [h, w, c_in], k: [c_out, kh, kw, c_in], b: [c_out] -> [h', w', c_out].
/// `&y + b` broadcasts right-aligned, so channels-last needs no `NewAxis`.
fn conv2d(x: &Ndarr<f32, U3>, k: &Ndarr<f32, U4>, b: &Ndarr<f32, U1>) -> Ndarr<f32, U3> {
    let (kh, kw) = (k.shape()[1], k.shape()[2]);
    let raw = x.slice(s![Win(kh), Win(kw), ..]).unwrap(); // [h', kh, w', kw, c]
    let patches = raw.permute_axes(&[0, 2, 1, 3, 4]).unwrap(); // [h', w', kh, kw, c]
    let filters = k.view().permute_axes(&[1, 2, 3, 0]).unwrap(); // [kh, kw, c, c_out]
    let y = patches
        .contract(&filters, U3::default(), |x, w| x * w, |a, b| a + b)
        .unwrap();
    (&y + b).map(|v| v.max(0.0)) // bias broadcast + ReLU
}

/// x: [h, w, c] -> [h/2, w/2, c]: windows, stride-2 positions, max-reduce.
fn max_pool_2x2(x: &Ndarr<f32, U3>) -> Ndarr<f32, U3> {
    let patches = x.slice(s![Win(2), Win(2), ..]).unwrap(); // [h', 2, w', 2, c]
    patches
        .slice(s![..;2, .., ..;2, .., ..])
        .unwrap() // [h/2, 2, w/2, 2, c]
        .reduce(3, f32::max)
        .unwrap()
        .reduce(1, f32::max)
        .unwrap()
}

fn shade(map: &Ndarr<f32, U2>) -> Ndarr<String, U2> {
    map.map(|v| {
        match v {
            v if *v <= 0.0 => "░░",
            v if *v < 2.0 => "▒▒",
            v if *v < 4.0 => "▓▓",
            _ => "██",
        }
        .to_string()
    })
}

fn main() {
    // A bright 10x12 rectangle on a dark 20x24 background, one channel.
    let mut pixels = vec![0.0_f32; 20 * 24];
    for row in 5..15 {
        for col in 6..18 {
            pixels[row * 24 + col] = 1.0;
        }
    }
    let image: Ndarr<f32, U3> = Ndarr::new(&pixels, [20, 24, 1]).unwrap();

    // Two Sobel detectors: left vertical edges and top horizontal edges.
    #[rustfmt::skip]
    let kernels = Ndarr::new(
        &[
            // gx: dark->bright left to right
            -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0_f32,
            // gy: dark->bright top to bottom
            -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0,
        ],
        [2, 3, 3, 1],
    )
    .unwrap();
    let bias = Ndarr::from([-0.5_f32, -0.5]); // small threshold before ReLU

    let features = conv2d(&image, &kernels, &bias); // [18, 22, 2]
    let pooled = max_pool_2x2(&features); // [9, 11, 2]

    for (name, channel) in [("vertical edges", 0isize), ("horizontal edges", 1)] {
        let full = features
            .slice(s![.., .., channel])
            .unwrap()
            .to_owned_array();
        let small = pooled.slice(s![.., .., channel]).unwrap().to_owned_array();
        println!("{name} {:?}:\n{}", features.shape(), shade(&full));
        println!("pooled {:?}:\n{}", pooled.shape(), shade(&small));
    }
}
