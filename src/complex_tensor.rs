use std::ops::{Add, Div, Mul, MulAssign, Neg};

use super::*;
use crate::complex::*;
use crate::scalars::Scalar;
use num_traits::{Float, Num};

impl Scalar for C<f64> {}
impl Scalar for C<f32> {}
impl Scalar for C<i128> {}
impl Scalar for C<i64> {}
impl Scalar for C<i32> {}
impl Scalar for C<i16> {}
impl Scalar for C<i8> {}
impl Scalar for C<isize> {}
impl Scalar for C<u128> {}
impl Scalar for C<u64> {}
impl Scalar for C<u32> {}
impl Scalar for C<u16> {}
impl Scalar for C<u8> {}
impl Scalar for C<usize> {}

impl<T: Copy + PartialEq + Clone + Debug + Default, R: Unsigned> Ndarr<C<T>, R> {
    pub fn re(&self) -> Ndarr<T, R> {
        let out = self.map_types(|z| z.re());
        out
    }
    pub fn im(&self) -> Ndarr<T, R> {
        let out = self.map_types(|z| z.im());
        out
    }
}

impl<T: Copy + PartialEq + Neg<Output = T> + Clone + Debug + Default, R: Unsigned>
    Ndarr<C<T>, R>
{
    /// Element wise complex conjugate.
    pub fn conj(&self) -> Self {
        let out = self.map(|z| z.conj());
        out
    }

    /// Conjugate or Hermitian transpose.
    pub fn conj_t(&self) -> Self {
        let out = self.map(|z| z.conj());
        out.t()
    }
}

impl<T, R: Unsigned> Ndarr<C<T>, R>
where
    T: Copy
        + PartialEq
        + Neg<Output = T>
        + Clone
        + Debug
        + Default
        + Div<Output = T>
        + Mul<Output = T>
        + Add<Output = T>,
{
    /// Applies inv element wise.
    pub fn inv(&self) -> Self {
        let out = self.map(|z| z.inv());
        out
    }
}

impl<
        T: Copy + PartialEq + Add<Output = T> + Mul<Output = T> + Clone + Debug + Default,
        R: Unsigned,
    > Ndarr<C<T>, R>
{
    pub fn r_square(&self) -> Ndarr<T, R> {
        let out = self.map_types(|z| z.r_square());
        out
    }
}

impl<T, R: Unsigned> Ndarr<C<T>, R>
where
    C<T>: MulAssign + Debug,
    T: Clone + Default + Debug + Copy + PartialEq + Num,
{
    pub fn powi(&self, n: i32) -> Self {
        let out = self.map(|z| z.powi(n));
        out
    }
}

//---------------Complex Float Tensors

impl<T, R: Unsigned> Ndarr<C<T>, R>
where
    T: Clone + Debug + Default + Float,
{
    pub fn abs(&self) -> Ndarr<T, R> {
        let out = self.map_types(|z| z.abs());
        out
    }
    pub fn exp(&self) -> Self {
        let out = self.map(|z| z.exp());
        out
    }
    pub fn arg(&self) -> Ndarr<T, R> {
        let out = self.map_types(|z| z.arg());
        out
    }
    pub fn ln(&self) -> Self {
        let out = self.map(|z| z.ln());
        out
    }
    pub fn sqrt(&self) -> Self {
        let out = self.map(|z| z.sqrt());
        out
    }
    pub fn powf(&self, n: T) -> Self {
        let out = self.map(|z| z.powf(n));
        out
    }
    pub fn powc(&self, _z: C<T>) -> Self {
        let out = self.map(|z| z.powc(*z));
        out
    }

    pub fn sin(&self) -> Self {
        let out = self.map(|z| z.sin());
        out
    }
    pub fn cos(&self) -> Self {
        let out = self.map(|z| z.cos());
        out
    }
    pub fn tan(&self) -> Self {
        let out = self.map(|z| z.tan());

        out
    }
    pub fn csc(&self) -> Self {
        let out = self.map(|z| z.csc());
        out
    }
    pub fn sec(&self) -> Self {
        let out = self.map(|z| z.sec());
        out
    }
    pub fn cot(&self) -> Self {
        let out = self.map(|z| z.cot());
        out
    }
    pub fn to_polar(&self) -> Ndarr<(T, T), R> {
        let out = self.map_types(|z| z.to_polar());
        out
    }

    pub fn is_infinite(&self) -> Ndarr<bool, R> {
        let out = self.map_types(|z| z.is_infinite());
        out
    }
    pub fn is_finite(&self) -> Ndarr<bool, R> {
        let out = self.map_types(|z| z.is_finite());
        out
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
        let out = self.map_types(|z| z.is_normal());
        out
    }
    pub fn is_nan(&self) -> Ndarr<bool, R> {
        let out = self.map_types(|z| z.is_nan());
        out
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
}
