use std::ops::{Add, Div, Mul, Neg};

use super::*;
use crate::complex::*;
use num_traits::{Float, Num};

impl<T: Scalar> Scalar for C<T> {}

// Element-wise complex ops accept any buffer/view and materialize owned outputs.
impl<T: Copy, R: Rank, B: Buffer<C<T>>> Ndarr<C<T>, R, B> {
    pub fn re(&self) -> Ndarr<T, R> {
        self.map(|z| z.re())
    }
    pub fn im(&self) -> Ndarr<T, R> {
        self.map(|z| z.im())
    }
}

impl<T: Copy + Num, R: Rank, B: Buffer<T>> Ndarr<T, R, B> {
    pub fn to_complex(&self) -> Ndarr<C<T>, R> {
        self.map(|x| C(*x, T::zero()))
    }
}

impl<T: Copy + Neg<Output = T>, R: Rank, B: Buffer<C<T>>> Ndarr<C<T>, R, B> {
    /// Element wise complex conjugate.
    pub fn conj(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.conj())
    }

    /// Conjugate or Hermitian transpose.
    pub fn h(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.conj()).t()
    }
}

impl<T, R: Rank, B: Buffer<C<T>>> Ndarr<C<T>, R, B>
where
    T: Copy + Neg<Output = T> + Div<Output = T> + Mul<Output = T> + Add<Output = T>,
{
    /// Element-wise `inv`.
    pub fn inv(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.inv())
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T>, R: Rank, B: Buffer<C<T>>> Ndarr<C<T>, R, B> {
    pub fn r_square(&self) -> Ndarr<T, R> {
        self.map(|z| z.r_square())
    }
}

impl<T: Copy + Num, R: Rank, B: Buffer<C<T>>> Ndarr<C<T>, R, B> {
    pub fn powi(&self, n: i32) -> Ndarr<C<T>, R> {
        self.map(|z| z.powi(n))
    }
}

impl<T: Float, R: Rank, B: Buffer<C<T>>> Ndarr<C<T>, R, B> {
    pub fn abs(&self) -> Ndarr<T, R> {
        self.map(|z| z.abs())
    }
    pub fn exp(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.exp())
    }
    pub fn arg(&self) -> Ndarr<T, R> {
        self.map(|z| z.arg())
    }
    pub fn ln(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.ln())
    }
    pub fn sqrt(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.sqrt())
    }
    pub fn powf(&self, n: T) -> Ndarr<C<T>, R> {
        self.map(|z| z.powf(n))
    }
    pub fn powc(&self, exponent: C<T>) -> Ndarr<C<T>, R> {
        self.map(|z| z.powc(exponent))
    }

    pub fn sin(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.sin())
    }
    pub fn cos(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.cos())
    }
    pub fn tan(&self) -> Ndarr<C<T>, R> {
        self.map(|z| z.tan())
    }
    pub fn to_polar(&self) -> Ndarr<(T, T), R> {
        self.map(|z| z.to_polar())
    }

    pub fn is_infinite(&self) -> Ndarr<bool, R> {
        self.map(|z| z.is_infinite())
    }
    pub fn is_finite(&self) -> Ndarr<bool, R> {
        self.map(|z| z.is_finite())
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
        self.map(|z| z.is_normal())
    }
    pub fn is_nan(&self) -> Ndarr<bool, R> {
        self.map(|z| z.is_nan())
    }
}

#[cfg(test)]
mod complex_tensor_test {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test() {
        let x = Ndarr::from([1, 2, 3]);
        let y = Ndarr::from([1.i(), 1.i(), 1.i()]);
        assert_eq!(&x + 1.i(), x + y);
    }

    #[test]
    fn exp_test() {
        let quads = Ndarr::from([1. + 0_f64.i(), 1.0.i(), -1. + 0_f64.i(), -1.0.i()]);
        println!("{:?}", quads * (PI / 2.).i().exp())
    }

    /// Both map families are generic over the buffer, so views call them on
    /// either side of the real/complex name collision.
    #[test]
    fn views_call_both_map_families() {
        let real = Ndarr::from([[0.0_f64, 1.0], [2.0, 3.0]]);
        assert_eq!(real.t_view().sin(), real.t().sin());
        assert_eq!(real.view().abs(), real.abs());
        let complex = real.to_complex();
        assert_eq!(complex.t_view().sin(), complex.t().sin());
        assert_eq!(complex.view().abs(), complex.abs());
    }
}
