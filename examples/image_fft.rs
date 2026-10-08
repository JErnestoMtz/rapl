//! Edge detection by frequency-domain convolution with a Laplacian kernel (I/O via the `image` crate).

use image::GrayImage;
use rapl::*;

fn main() {
    // Grayscale f32 in [0, 1], row-major: shape [height, width].
    let img = image::open("graphics/peppers.png").unwrap().to_luma32f();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let img: Ndarr<f32, U2> = Ndarr::new(img.as_raw(), [h, w]).unwrap();

    let fft = img.to_complex().fft2d();

    // A centered discrete Laplacian kernel.
    let mut kernel: Ndarr<f32, U2> = Ndarr::zeros([h, w]);
    let (mid_r, mid_c) = (h / 2, w / 2);
    kernel[[mid_r, mid_c]] = 4.;
    kernel[[mid_r + 1, mid_c]] = -1.;
    kernel[[mid_r - 1, mid_c]] = -1.;
    kernel[[mid_r, mid_c + 1]] = -1.;
    kernel[[mid_r, mid_c - 1]] = -1.;

    // Multiply in the frequency domain, invert, and recenter: convolution.
    let out = (fft * kernel.to_complex().fft2d()).ifft2().fftshift().re();

    let min = out.iter_elems().cloned().reduce(f32::min).unwrap();
    let max = out.iter_elems().cloned().reduce(f32::max).unwrap();
    let pixels: Vec<u8> = out
        .data()
        .iter()
        .map(|v| ((v - min) / (max - min) * 255.0) as u8)
        .collect();
    GrayImage::from_raw(w as u32, h as u32, pixels)
        .unwrap()
        .save("graphics/pepper_edges.png")
        .unwrap();
}
