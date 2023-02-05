use crate::scalars::Scalar;

use super::*;
use std::ops::*;

// Arithmetic operations

impl<P,T, const R: usize, const N: usize> Add<P> for Ndarr<T, N, R>
where
    T: Add<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
    P: IntoNdarr<T,N,R>,
{
    type Output = Self;
    fn add(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self);
            self.bimap(other, |x, y| *x + *y)
    }
}

impl<P,T, const R: usize, const N: usize> Sub<P> for Ndarr<T, N, R>
where
    T: Sub<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
    P: IntoNdarr<T,N,R>,
{
    type Output = Self;
    fn sub(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x - *y)
        }
    }
}



impl<P,T, const R: usize, const N: usize> Mul<P> for Ndarr<T, N, R>
where
    T: Mul<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
    P: IntoNdarr<T,N,R>,
{
    type Output = Self;
    fn mul(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x * *y)
        }
    }
}

impl<P,T, const R: usize, const N: usize> Div<P> for Ndarr<T, N, R>
where
    T: Div<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
    P: IntoNdarr<T,N,R>,
{
    type Output = Self;
    fn div(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x / *y)
        }
    }
}
impl<P,T, const R: usize, const N: usize> Rem<P> for Ndarr<T, N, R>
where
    T: Rem<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
    P: IntoNdarr<T,N,R>,
{
    type Output = Self;
    fn rem(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations

        let other = other.into_ndarr(&self);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x % *y)
        }
    }
}

impl<T, const R: usize, const N: usize> Neg for Ndarr<T, N, R>
where
    T: Neg<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }

}

// Assign traits

impl<P, T, const R: usize, const N: usize> AddAssign<P> for Ndarr<T, N, R>
where 
    T: Add<Output = T> + Copy + Clone + Debug + Default,
    P: Into<T> + Copy,
    [T; N]: Default,
{
    //TODO: to be more general es better to converte P into Ndarr<T,N,R> and then use bimap in place. but first we need the casting trait
    fn add_assign(&mut self, other: P){
        self.mapinplace(|x| *x + other.into())
    }
}