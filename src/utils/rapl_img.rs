use super::Ndarr;

use image::*;

use crate::shape::Dim;
use std::path::Path;
use typenum::{U2, U3};

pub use image::ImageFormat;

/// Open an image as RGB represented as  `Ndarr<u8,3>` where the axis dimensions
/// are (width, height, 3), were the depth represent each color channel.
pub fn open_rgbu8<P: AsRef<Path>>(path: P) -> Result<Ndarr<u8, U3>, ImageError> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    let _n = w as usize * h as usize * 3;

    let data_arr = img.iter().copied().collect();
    Ok(Ndarr {
        data: data_arr,
        dim: Dim::new(&[w as usize, h as usize, 3]).unwrap(),
    })
}

/// Open an image as RGB represented as  `Ndarr<f32,3>` where the axis dimensions
/// are (width, height, 3), were the depth represent each color channel.
/// Each subpixel has a value from 0 to 1.
pub fn open_rgbf32<P: AsRef<Path>>(path: P) -> Result<Ndarr<f32, U3>, ImageError> {
    //open image transform to rgb8
    let img = image::open(path)?.to_rgb32f();
    let (w, h) = (img.width(), img.height());
    let _n = w as usize * h as usize * 3;

    let data_arr = img.iter().copied().collect();
    Ok(Ndarr {
        data: data_arr,
        dim: Dim::new(&[w as usize, h as usize, 3]).unwrap(),
    })
}

///Open an image as Luma (black and white) represented as  `Ndarr<u8,2>` where the axis dimensions are (width, height).
pub fn open_lumau8<P: AsRef<Path>>(path: P) -> Result<Ndarr<u8, U2>, ImageError> {
    let img = image::open(path)?.to_luma8();
    let (w, h) = (img.width(), img.height());
    let _n = w as usize * h as usize;

    let data_arr = img.iter().copied().collect();
    Ok(Ndarr {
        data: data_arr,
        dim: Dim::new(&[w as usize, h as usize]).unwrap(),
    })
}

/// Open an image as Luma (black and white) represented as  `Ndarr<f32,2>`
/// where the axis dimensions are (width, height).
/// Each pixel is detonted by a f32 from 0.0 to 1.0.
pub fn open_lumaf32<P: AsRef<Path>>(path: P) -> Result<Ndarr<f32, U2>, ImageError> {
    let img = image::open(path)?.to_luma32f();
    let (w, h) = (img.width(), img.height());
    let _n = w as usize * h as usize;

    let data_arr = img.iter().copied().collect();
    Ok(Ndarr {
        data: data_arr,
        dim: Dim::new(&[w as usize, h as usize]).unwrap(),
    })
}

impl Ndarr<u8, U3> {
    /// Saves a Ndarr<u8,3> with shape (with, heighth, 3) as RGB Image. Takes
    /// path and format, where format is enum: `ImageFormat`.
    pub fn save_as_rgb<P: AsRef<Path>>(&self, path: P, fmt: ImageFormat) {
        assert!(
            self.dim.shape[2] == 3,
            "Cannot convert Ndarr of shape {:?} to RGB image",
            self.dim.shape
        );
        let w = self.dim.shape[0];
        let h = self.dim.shape[1];
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(w as u32, h as u32, self.data.clone())
                .expect("Could not create ImageBuffer with Ndarr");
        img.save_with_format(path, fmt)
            .expect("Error, saving array as Image");
    }
}

impl Ndarr<f32, U3> {
    /// Saves a Ndarr<f32,3> with shape (with, heighth, 3) as RGB Image. Takes
    /// path and format, where format is enum: `ImageFormat`.
    pub fn save_as_rgb<P: AsRef<Path>>(&self, path: P, fmt: ImageFormat) {
        assert!(
            self.dim.shape[2] == 3,
            "Cannot convert Ndarr of shape {:?} to RGB image",
            self.dim.shape
        );
        let w = self.dim.shape[0];
        let h = self.dim.shape[1];
        let img: ImageBuffer<Rgb<f32>, Vec<f32>> =
            ImageBuffer::from_raw(w as u32, h as u32, self.data.clone())
                .expect("Could not create ImageBuffer with Ndarr");
        img.save_with_format(path, fmt)
            .expect("Error, saving array as Image");
    }
}

impl Ndarr<u8, U2> {
    /// Saves a Ndarr<u8,2> with shape (with, heighth) as Luma (Black and white)
    /// Image. Takes path and format, where format is enum: `ImageFormat`.
    pub fn save_as_luma<P: AsRef<Path>>(&self, path: P, fmt: ImageFormat) {
        let w = self.dim.shape[0];
        let h = self.dim.shape[1];
        let img: ImageBuffer<Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(w as u32, h as u32, self.data.clone())
                .expect("Could not create ImageBuffer with Ndarr");
        img.save_with_format(path, fmt)
            .expect("Error, saving array as Image");
    }
}

impl Ndarr<f32, U2> {
    /// Normalize a Ndarr<f32,2> to values form 0.0 to 1.0 and saves it as
    /// Luma (Black and white) Image. Takes path and format, where format
    /// is enum: `ImageFormat`.
    pub fn save_as_luma<P: AsRef<Path>>(&self, path: P, fmt: ImageFormat) {
        let max = self
            .data
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .expect("Cannot convert empty Ndarr to Image");
        let min = self
            .data
            .iter()
            .min_by(|a, b| a.total_cmp(b))
            .expect("Cannot convert empty Ndarr to Image");

        let norm_arr = (self - min) / (max - min);
        let w = self.dim.shape[0];
        let h = self.dim.shape[1];
        let im_u8 = norm_arr.map(|x| (*x * u16::MAX as f32) as u16);
        let img: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_raw(w as u32, h as u32, im_u8.data)
                .expect("Could not create ImageBuffer with Ndarr");
        img.save_with_format(path, fmt)
            .expect("Error, saving array as Image");
    }
}

#[cfg(test)]
mod image_test {
    use super::*;
    use crate::de_slice;
    #[test]
    fn open_rgb8() {
        let img = open_rgbu8("graphics/test_img.jpg").unwrap();
        let mut slices = img.slice_at(2);
        slices[2].map_in_place(|x| x.wrapping_add(200));
        let des = de_slice(&slices, 2);
        des.save_as_rgb("graphics/out_blue.png", ImageFormat::Png);
    }
    #[test]
    fn open_f32() {
        let img = open_lumaf32("graphics/test_img.jpg").unwrap();
        img.save_as_luma("graphics/out_test_bw.jpg", ImageFormat::Png);
        //square image
        let square = &img * &img;
        square.save_as_luma("graphics/out_test_bw_square.jpg", ImageFormat::Png);
    }
}
