use crate::scalars::Scalar;

use super::*;
use std::ops::*;

// Arithmetic operations

impl<P, T, const R: usize> Add<P> for Ndarr<T, R>
where
    T: Add<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R>,
{
    type Output = Self;
    fn add(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.bimap(other, |x, y| *x + *y)
    }
}
impl<P, T, const R: usize> Sub<P> for Ndarr<T, R>
where
    T: Sub<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R>,
{
    type Output = Self;
    fn sub(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x - *y)
        }
    }
}

impl<P, T, const R: usize> Mul<P> for Ndarr<T, R>
where
    T: Mul<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R>,
{
    type Output = Self;
    fn mul(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x * *y)
        }
    }
}

impl<P, T, const R: usize> Div<P> for Ndarr<T, R>
where
    T: Div<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R>,
{
    type Output = Self;
    fn div(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x / *y)
        }
    }
}
impl<P, T, const R: usize> Rem<P> for Ndarr<T, R>
where
    T: Rem<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R>,
{
    type Output = Self;
    fn rem(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.bimap(other, |x, y| *x % *y)
        }
    }
}

impl<T, const R: usize> Neg for Ndarr<T, R>
where
    T: Neg<Output = T> + Copy + Clone + Debug + Default,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}

// Assign traits

impl<P, T, const R: usize> AddAssign<P> for Ndarr<T, R>
where
    T: Add<Output = T> + Copy + Clone + Debug + Default,
    P: Into<T> + Copy,
{
    //TODO: to be more general es better to converted P into Ndarr<T,N,R> and then use bimap in place. but first we need the casting trait
    fn add_assign(&mut self, other: P) {
        self.map_in_place(|x| *x + other.into())
    }
}

///////////////////////////// As references

impl<P, T, const R: usize> Add<&P> for &Ndarr<T, R>
where
    T: Add<Output = T> + Copy + Clone + Debug + Default,
    P: Clone + IntoNdarr<T, R>,
{
    type Output = Ndarr<T, R>;
    fn add(self, other: &P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.clone().into_ndarr(&self.shape);
        self.clone().bimap(other, |x, y| *x + *y)
    }
}


impl<P, T, const R: usize> Sub<&P> for &Ndarr<T, R>
where
    T: Sub<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    type Output = Ndarr<T, R>;
    fn sub(self, other: &P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.clone().into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.clone().bimap(other, |x, y| *x - *y)
        }
    }
}

impl<P, T, const R: usize> Mul<&P> for &Ndarr<T, R>
where
    T: Mul<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    type Output = Ndarr<T,R>;
    fn mul(self, other: &P) -> Self::Output{
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.clone().into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.clone().bimap(other, |x, y| *x * *y)
        }
    }
}

impl<P, T, const R: usize> Div<&P> for &Ndarr<T, R>
where
    T: Div<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    type Output = Ndarr<T,R>;
    fn div(self, other: &P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.clone().into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.clone().bimap(other, |x, y| *x / *y)
        }
    }
}
impl<P, T, const R: usize> Rem<&P> for &Ndarr<T, R>
where
    T: Rem<Output = T> + Copy + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    type Output = Ndarr<T,R>;
    fn rem(self, other: &P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.clone().into_ndarr(&self.shape);
        if self.shape != other.shape {
            panic!("Shape mismatch")
        } else {
            self.clone().bimap(other, |x, y| *x % *y)
        }
    }
}

impl<T, const R: usize> Neg for &Ndarr<T, R>
where
    T: Neg<Output = T> + Copy + Clone + Debug + Default,
{
    type Output = Ndarr<T,R>;
    fn neg(self) -> Self::Output {
        self.clone().map(|x| -*x)
    }
}

// Assign traits
