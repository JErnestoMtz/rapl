use crate::helpers::multiply_list;

use super::*;

// this is a work around to not conflict between scalars and Ndarr for IntoNdarr trait,
//TODO: think of a better solution
pub trait Scalar {}
impl Scalar for f64 {}
impl Scalar for f32 {}
impl Scalar for i64 {}
impl Scalar for i32 {}
impl Scalar for i16 {}
impl Scalar for i8 {}
impl Scalar for isize {}
impl Scalar for u64 {}
impl Scalar for u32 {}
impl Scalar for u16 {}
impl Scalar for u8 {}
impl Scalar for usize {}
impl Scalar for char {}
impl Scalar for &str {}


pub fn extend_scalar<P, T, const R: usize>(scalar: P, shape: &[usize; R]) -> Ndarr<T, R>
where
    T: Debug + Copy + Clone + Default,
    P: Into<T> + Clone,
{
    let n = multiply_list(shape, 1);
    let s = shape.clone();
    let mut out_data = vec![T::default(); n];
    for i in 0..out_data.len() {
        out_data[i] = scalar.clone().into();
    }
    Ndarr {
        data: out_data,
        shape: s,
    }
}

impl<T, P, const R: usize> IntoNdarr<T, R> for P
where
    T: Debug + Copy + Clone + Default,
    P: Into<T> + Clone + Scalar,
{
    fn into_ndarr(self, shape: &[usize; R]) -> Ndarr<T, R> {
        extend_scalar(self, shape)
    }
    fn get_rank(&self) -> usize {
        0
    }
}

pub trait Trig {
    fn f_sin(&self) -> Self;
    fn f_cos(&self) -> Self;
    fn f_tan(&self) -> Self;
    fn f_sinh(&self) -> Self;
    fn f_cosh(&self) -> Self;
    fn f_tanh(&self) -> Self;
}

impl Trig for f32 {
    fn f_sin(&self) -> Self {
        self.sin()
    }
    fn f_cos(&self) -> Self {
        self.cos()
    }
    fn f_tan(&self) -> Self {
        self.tan()
    }
    fn f_sinh(&self) -> Self {
        self.sinh()
    }
    fn f_cosh(&self) -> Self {
        self.cosh()
    }
    fn f_tanh(&self) -> Self {
        self.tanh()
    }
}

impl Trig for f64 {
    fn f_sin(&self) -> Self {
        self.sin()
    }
    fn f_cos(&self) -> Self {
        self.cos()
    }
    fn f_tan(&self) -> Self {
        self.tan()
    }
    fn f_sinh(&self) -> Self {
        self.sinh()
    }
    fn f_cosh(&self) -> Self {
        self.cosh()
    }
    fn f_tanh(&self) -> Self {
        self.tanh()
    }
}

