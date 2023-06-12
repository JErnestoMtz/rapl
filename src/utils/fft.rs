use super::{de_slice, Ndarr, C, U1, U2};
use core::fmt::Debug;
use num_traits::{FromPrimitive, Num, Signed};
use rustfft::{num_complex::Complex, FftPlanner};
use std::marker::{Send, Sync};

//rustfft use cum_complex so this functions are to translate num_complex to rapl complex
fn to_numcomplex<T: Copy + PartialEq>(c: C<T>) -> Complex<T> {
    Complex { re: c.0, im: c.1 }
}

fn to_raplcomplex<T: Copy + PartialEq>(c: Complex<T>) -> C<T> {
    C(c.re, c.im)
}

impl<T: Clone + Debug + Copy + PartialEq> Ndarr<C<T>, U1> {
    ///Performs the one dimensional Fourier Transform to a rank one rapl array: `Ndarr<C<T>,U1>`.
    pub fn fft(&self) -> Ndarr<C<T>, U1>
    where
        T: Signed + FromPrimitive + Num + Copy + PartialEq + Send + Sync + 'static,
    {
        let mut buff: Vec<Complex<T>> = self.data.iter().map(|x| to_numcomplex(*x)).collect();
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(self.len());
        fft.process(&mut buff);
        let data: Vec<C<T>> = buff.iter().map(|x| to_raplcomplex(*x)).collect();
        Ndarr {
            data,
            dim: self.dim.clone(),
        }
    }

    ///Performs the one dimensional Inverse Fourier Transform to a rank one rapl array: `a = Ndarr<C<T>,U1>`.
    /// then `a.fft().ifft()` is approximately equal to `a`
    pub fn ifft(&self) -> Ndarr<C<T>, U1>
    where
        T: Signed + FromPrimitive + Num + Copy + PartialEq + Send + Sync + 'static,
    {
        let n = self.len();
        let n_t = T::from_usize(n).unwrap();
        let mut buff: Vec<Complex<T>> = self.data.iter().map(|x| to_numcomplex(*x)).collect();
        let mut planner = FftPlanner::<T>::new();
        //this does not compute the actual jifft, it just fft in the inverse direction thats why we need the normalization
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut buff);
        let data: Vec<C<T>> = buff.iter().map(|c| C(c.re / n_t, c.im / n_t)).collect();
        Ndarr {
            data,
            dim: self.dim.clone(),
        }
    }
}

impl<T: Clone + Debug + Copy + PartialEq> Ndarr<C<T>, U2> {
    ///Performs the two dimensional Fourier Transform to a rank two rapl array: `Ndarr<C<T>,U2>`.
    pub fn fft2d(&self) -> Ndarr<C<T>, U2>
    where
        T: Signed + FromPrimitive + Num + Copy + PartialEq + Send + Sync + 'static,
    {
        //A bit ugly but faster that using the 1d fft becouse we can reuse fftPlaner
        let arr = self.map(|x| to_numcomplex(*x));
        let mut planner = FftPlanner::<T>::new();
        let n_x = self.shape()[0];
        let n_y = self.shape()[1];
        let fft_x = planner.plan_fft_forward(n_x);
        let fft_y = planner.plan_fft_forward(n_y);
        let mut vec_x: Vec<Vec<Complex<T>>> =
            arr.slice_at(1).iter().map(|x| x.data.clone()).collect();
        for v in vec_x.iter_mut() {
            fft_x.process(v)
        }
        let vec_x: Vec<Ndarr<Complex<T>, U1>> = vec_x
            .iter()
            .map(|v| Ndarr::new(v, [n_x]).unwrap())
            .collect();
        let x_pass = de_slice(&vec_x, 1);
        let mut vec_y: Vec<Vec<Complex<T>>> =
            x_pass.slice_at(0).iter().map(|x| x.data.clone()).collect();
        for v in vec_y.iter_mut() {
            fft_y.process(v)
        }
        let vec_y: Vec<Ndarr<Complex<T>, U1>> = vec_y
            .iter()
            .map(|v| Ndarr::new(v, [n_y]).unwrap())
            .collect();
        de_slice(&vec_y, 0).map(|x| to_raplcomplex(*x))
    }

    pub fn ifft2(&self) -> Ndarr<C<T>, U2>
    where
        T: Signed + FromPrimitive + Num + Copy + PartialEq + Send + Sync + 'static,
    {
        //A bit ugly but faster that using the 1d fft becouse we can reuse fftPlaner
        let arr = self.map(|x| to_numcomplex(*x));
        let mut planner = FftPlanner::<T>::new();
        let n_x = self.shape()[0];
        let n_y = self.shape()[1];

        let factor = T::from_usize(n_x * n_y).unwrap();

        let fft_x = planner.plan_fft_inverse(n_x);
        let fft_y = planner.plan_fft_inverse(n_y);
        let mut vec_y: Vec<Vec<Complex<T>>> =
            arr.slice_at(0).iter().map(|y| y.data.clone()).collect();
        for v in vec_y.iter_mut() {
            fft_y.process(v)
        }
        let vec_y: Vec<Ndarr<Complex<T>, U1>> = vec_y
            .iter()
            .map(|v| Ndarr::new(v, [n_y]).unwrap())
            .collect();
        let y_pass = de_slice(&vec_y, 0);
        let mut vec_x: Vec<Vec<Complex<T>>> =
            y_pass.slice_at(1).iter().map(|x| x.data.clone()).collect();
        for v in vec_x.iter_mut() {
            fft_x.process(v)
        }
        let vec_x: Vec<Ndarr<Complex<T>, U1>> = vec_x
            .iter()
            .map(|v| Ndarr::new(v, [n_x]).unwrap())
            .collect();
        de_slice(&vec_x, 1).map(|c| C(c.re / factor, c.im / factor))
    }
}

impl<T: Clone + Debug> Ndarr<T, U1> {
    pub fn fftshif(&self) -> Self {
        let n = self.len();
        match n % 2 {
            0 => {
                let mut new_data: Vec<T> = Vec::with_capacity(n);
                new_data.extend_from_slice(&self.data[n / 2..]);
                new_data.extend_from_slice(&self.data[..n / 2]);
                Ndarr {
                    data: new_data,
                    dim: self.dim.clone(),
                }
            }
            _ => {
                let mut new_data: Vec<T> = Vec::with_capacity(n);
                new_data.extend_from_slice(&self.data[(n + 1) / 2..]);
                new_data.extend_from_slice(&self.data[..(n + 1) / 2]);
                Ndarr {
                    data: new_data,
                    dim: self.dim.clone(),
                }
            }
        }
    }
}

impl<T: Clone + Debug> Ndarr<T, U2> {
    pub fn fftshif(&self) -> Self {
        let (m, n) = (self.shape()[0], self.shape()[1]);
        let mut new_data = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let row = i / n;
            let col = i % n;
            let shifted_row = (row + (m + 1) / 2) % m;
            let shifted_col = (col + (n + 1) / 2) % n;
            let shifted_index = shifted_row * n + shifted_col;
            new_data.push(self.data[shifted_index].clone());
        }
        Ndarr {
            data: new_data,
            dim: self.dim.clone(),
        }
    }
}
#[cfg(test)]
mod fft_test {
    use super::*;
    #[test]
    fn test_1d() {
        let a = Ndarr::from([0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, -0.1, 0.2]).to_complex();
        let fft_a_numpy: Ndarr<C<f64>, U1> = Ndarr::from([
            C(0.7, 0.0),
            C(0.26244852, -0.14456102),
            C(0.19606372, -0.09072781),
            C(-0.05, 0.08660254),
            C(-0.30851223, 0.31364084),
            C(-0.30851223, -0.31364084),
            C(-0.05, -0.08660254),
            C(0.19606372, 0.09072781),
            C(0.26244852, 0.14456102),
        ]);
        let rapl_fft = a.fft();

        //test fft
        assert!(rapl_fft.re().approx(&fft_a_numpy.re()));
        assert!(rapl_fft.im().approx(&fft_a_numpy.im()));
        //test ifft
        assert!(rapl_fft.ifft().re().approx(&a.re()));
        assert!(rapl_fft.ifft().im().approx(&a.im()));
    }
    #[test]
    fn test_2d() {
        let a = Ndarr::from([0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, -0.1, 0.2])
            .to_complex()
            .reshape([3, 3])
            .unwrap();
        let numpy_fft2: Ndarr<C<f64>, U2> = Ndarr::from([
            [C(0.7, 0.), C(-0.05, 0.08660254), C(-0.05, -0.08660254)],
            [
                C(0.25, 0.08660254),
                C(-0.35, -0.08660254),
                C(0.25, 0.25980762),
            ],
            [
                C(0.25, 0.08660254),
                C(0.25, -0.25980762),
                C(-0.35, 0.08660254),
            ],
        ]);
        let rapl_fft2 = a.fft2d();
        //test fft
        assert!(rapl_fft2.re().approx(&numpy_fft2.re()));
        assert!(rapl_fft2.im().approx(&numpy_fft2.im()));
        //test ifft
        assert!(rapl_fft2.ifft2().re().approx(&a.re()));
        assert!(rapl_fft2.ifft2().im().approx(&a.im()));
    }

    #[test]
    fn fftshif_1d() {
        let odd = Ndarr::from([1, 2, 3, 4, 5, 6, 7]);
        let pair = Ndarr::from([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(odd.fftshif(), Ndarr::from([5, 6, 7, 1, 2, 3, 4]));
        assert_eq!(pair.fftshif(), Ndarr::from([5, 6, 7, 8, 1, 2, 3, 4]));
    }

    #[test]
    fn fftshif_2d() {
        let odd = Ndarr::from(0..9).reshape([3, 3]).unwrap();
        let pair = Ndarr::from(0..16).reshape([4, 4]).unwrap();
        let odd_p = Ndarr::from(0..12).reshape([3, 4]).unwrap();
        assert_eq!(
            odd.fftshif(),
            Ndarr::from([[8, 6, 7], [2, 0, 1], [5, 3, 4]])
        );
        assert_eq!(
            pair.fftshif(),
            Ndarr::from([[10, 11, 8, 9], [14, 15, 12, 13], [2, 3, 0, 1], [6, 7, 4, 5]])
        );
        println!("{}", odd_p.fftshif());
    }
}
