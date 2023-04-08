use std::{ops::{Neg, Mul, Add}, clone, f64::consts::PI};

use super::*;
use crate::complex::*;
use num_traits::Float;
use crate::scalars::Scalar;

    
impl Scalar for C<f64> {}
impl Scalar for C<f32> {}
impl Scalar for C<i128> {}
impl Scalar for C<i64> {}
impl Scalar for C<i32> {}
impl Scalar for C<i16> {}
impl Scalar for C<i8 >{}
impl Scalar for C<isize> {}
impl Scalar for C<u128> {}
impl Scalar for C<u64> {}
impl Scalar for C<u32> {}
impl Scalar for C<u16> {}
impl Scalar for C<u8>{}
impl Scalar for C<usize> {}

impl<T: Copy + PartialEq + Clone + Debug + Default, const R: usize> Ndarr<C<T>,R>{
    pub fn re(&self) -> Ndarr<T,R>{
        let out = self.clone().map_types(|z| z.re());
        out
    }
    pub fn im(&self) -> Ndarr<T,R> {
        let out = self.clone().map_types(|z| z.im());
        out
    }
}

impl<T: Copy + PartialEq + Neg<Output = T> + Clone + Debug + Default, const R: usize> Ndarr<C<T>,R> {
    pub fn conj(&self) -> Self {
        let out = self.clone().map(|z| z.conj());
        out
    }
}

impl<T: Copy + PartialEq + Add<Output = T> + Mul<Output = T>+ Clone + Debug + Default , const R: usize> Ndarr<C<T>,R> {
    pub fn r_square(&self) -> Ndarr<T,R> {
        let out = self.clone().map_types(|z| z.r_square());
        out
    }
}



impl <T, const R: usize> Ndarr<C<T>,R> 
where T: Clone + Debug + Default + Float
{

    pub fn abs(&self)->Ndarr<T, R>{
       let out = self.clone().map_types(|z| z.abs() );
       out
    }
    pub fn exp(&self)->Self{
       let out = self.clone().map(|z| z.exp() );
       out
    }
    pub fn arg(&self)->Ndarr<T,R>{
       let out = self.clone().map_types(|z| z.arg() );
       out
    }
    pub fn ln(&self)->Self{
       let out = self.clone().map(|z| z.ln() );
       out
    }
    pub fn sin(&self)->Self{
       let out = self.clone().map(|z| z.sin() );
       out
    }
    pub fn cos(&self)->Self{
       let out = self.clone().map(|z| z.cos() );
       out
    }
    pub fn tan(&self)->Self{
       let out = self.clone().map(|z| z.tan() );

       out
    }
    pub fn csc(&self)->Self{
       let out = self.clone().map(|z| z.csc() );
       out
    }
    pub fn sec(&self)->Self{
       let out = self.clone().map(|z| z.sec() );
       out
    }
    pub fn cot(&self)->Self{
       let out = self.clone().map(|z| z.cot() );
       out
    }

    pub fn to_polar(&self) -> Ndarr<(T, T),R> {
       let out = self.clone().map_types(|z| z.to_polar() );
       out
    }

    pub fn is_infinite(&self) -> Ndarr<bool,R> {
       let out = self.clone().map_types(|z| z.is_infinite() );
       out
    }
    pub fn is_finite(&self) -> Ndarr<bool,R> {
       let out = self.clone().map_types(|z| z.is_finite() );
       out
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
       let out = self.clone().map_types(|z| z.is_normal() );
       out

    }
    pub fn is_nan(&self) -> Ndarr<bool,R>{
       let out = self.clone().map_types(|z| z.is_nan() );
       out
    }

}

#[cfg(test)]
mod complex_tensor_test{
    use super::*;
    #[test]
    fn exp_test(){
        let quads =  Ndarr::from([1. + 0_f64.i(), 1.0.i(), -1. + 0_f64.i(), -1.0.i()]);
        println!("{:?}",  quads * (PI/2.).i().exp() )
    }
}