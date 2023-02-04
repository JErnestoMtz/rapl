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



pub fn extend_scalar<P,T, const N: usize, const R: usize>(scalar: P, ndarr: &Ndarr<T, N, R>)-> Ndarr<T,N,R>
    where T: Debug + Copy + Clone + Default,
    P: Into<T> + Clone,
    [T; N]: Default
{
   let mut out_data: [T; N] = Default::default();
   for i in 0..N{
    out_data[i] = scalar.clone().into();
   };
   Ndarr { data: out_data, shape: ndarr.shape.clone() }

}

//TODO: the problem her is we can not use Into because we need to know the shape, and Into trait does not passes any reference 
trait IntoNdarr<T, const N: usize, const R: usize>
    where T: Debug + Copy + Clone + Default,
    [T; N]: Default
{
    fn into_ndarr(self, ndarr: &Ndarr<T,N, R>) -> Ndarr<T,N,R>;
}


impl<T, P, const N: usize, const R: usize> IntoNdarr<T,N,R> for P
    where T: Debug + Copy + Clone + Default,
    P: Into<T> + Clone + Scalar,
    [T; N]: Default
{
    fn into_ndarr(self, ndarr: &Ndarr<T,N, R>) -> Ndarr<T,N,R> {
        extend_scalar(self, ndarr)
    }
}

impl<T, const N: usize, const R: usize> IntoNdarr<T,N,R> for Ndarr<T,N,R>
    where T: Debug + Copy + Clone + Default,
    [T; N]: Default
{
    fn into_ndarr(self, ndarr: &Ndarr<T,N, R>) -> Ndarr<T,N,R> {
        self
    }
}