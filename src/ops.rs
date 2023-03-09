use crate::{scalars::Scalar, primitives::{Broadcast, DimError, Reduce, Broadcast_data}, helpers::{const_max}};

use super::*;
use std::ops::*;


// Arithmetic operations

impl<P, T, const R: usize> Add<P> for Ndarr<T, R>
where
    T: Add<Output = T> + Copy + Clone + Debug + Default + Scalar,
    P: IntoNdarr<T, R>,
{
    type Output = Self;
    fn add(self, other: P) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.bimap(other, |x, y| *x + *y)
    }
}

//TODO: this is super cursed but is the only solution I found 
pub fn poly_diatic<F,T1,T2,T3, const R1: usize, const R2: usize>(arr1: Ndarr<T1,R1>, arr2: Ndarr<T2,R2>, f: F)->Result<Ndarr<T3,{const_max(R1,  R2)}>,DimError>
where
    T1: Copy + Clone + Debug + Default,
    T2: Copy + Clone + Debug + Default,
    T3: Copy + Clone + Debug + Default,
    F: Fn(T1,T2) -> T3,
    [usize; {const_max(R2, R1)}]: Sized,
{
    let new_shape = helpers::broadcast_shape(&arr1.shape, &arr2.shape)?;
    let cast1 = arr1.broadcast(&arr2.shape)?; 
    let cast2 = arr2.broadcast(&arr1.shape)?;
    let mut new_data = vec![T3::default(); cast2.len()];
    for i in 0..cast1.len(){
        new_data[i] = f(cast1.data[i], cast2.data[i])
    }
    return Ok(Ndarr { data: new_data , shape: new_shape });
}


//TODO: found some way to simplify, this has concerning levels of cursedness!!

pub fn mat_mul<T,const R1: usize, const R2: usize>(arr1: Ndarr<T,R1>, arr2: Ndarr<T,R2>)->Ndarr<T,{const_max(R1 + R2 - 1, R1 + R2 - 1)-1}>
    where T: Sub<Output = T> + Copy + Clone + Debug + Default + Add<Output = T> + Mul<Output = T>+Display,
    [usize; const_max(R1, R1 + R2 - 1)]: Sized,
    [usize; const_max(R2, R2 + R1 - 1)]: Sized, //BUG: Actually a bug with rust compiler that doesn't idintyfy permutations of arithmetic operation.
    [usize; const_max(R2, R1 + R2 - 1)]: Sized,
    [usize; const_max(R1 + R2 - 1, R1 + R2 - 1)]: Sized, //same here
    [usize; const_max(R1 + R2 - 1, R2 + R1 - 1)]: Sized,
    [usize; const_max(R1 + R2 - 1, R1 + R2 - 1)-1]: Sized, //same here
    [usize; const_max(R1, R2)]: Sized,

{
    let arr1 = arr1.t();
    let padded1: [usize; R1 + R2 -1] = helpers::path_shape(&arr1.shape).unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr{data: bdata, shape: padded1}.t();
    let padded2: [usize; R1+ R2 -1] = helpers::path_shape(&arr2.shape).unwrap();
    let bdata2 = arr2.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr{data: bdata2, shape: padded2};
    let r = poly_diatic(arr1, arr2, |x,y| x*y).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
        //seems to work for all test I did
    let rr = r.reduce(R1-1, |x,y| *x+*y).unwrap();
    return rr
}


fn inner<F,G,T1,T2,T3,const R1: usize, const R2: usize>(f: F, g: G, arr1: Ndarr<T1,R1>, arr2: Ndarr<T2,R2>)->Ndarr<T3,{const_max(R1 + R2 - 1, R1 + R2 - 1)-1}>
    where 
    T1: Copy + Clone + Debug + Default,
    T2: Copy + Clone + Debug + Default,
    T3: Copy + Clone + Debug + Default,
    [usize; const_max(R1, R1 + R2 - 1)]: Sized,
    [usize; const_max(R2, R2 + R1 - 1)]: Sized, //BUG: Actually a bug with rust compiler that doesn't idintyfy permutations of arithmetic operation.
    [usize; const_max(R2, R1 + R2 - 1)]: Sized,
    [usize; const_max(R1 + R2 - 1, R1 + R2 - 1)]: Sized, //same here
    [usize; const_max(R1 + R2 - 1, R2 + R1 - 1)]: Sized,
    [usize; const_max(R1 + R2 - 1, R1 + R2 - 1)-1]: Sized, //same here
    [usize; const_max(R1, R2)]: Sized,
    F: Fn(T1,T2)->T3,
    G: Fn(T3,T3)->T3


{
    let arr1 = arr1.t();
    let padded1: [usize; R1 + R2 -1] = helpers::path_shape(&arr1.shape).unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr{data: bdata, shape: padded1}.t();
    let padded2: [usize; R1+ R2 -1] = helpers::path_shape(&arr2.shape).unwrap();
    let bdata2 = arr2.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr{data: bdata2, shape: padded2};
    let r = poly_diatic(arr1, arr2, f).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
        //seems to work for all test I did
    let rr = r.reduce(R1-1, |x,y| g(*x,*y)).unwrap();
    return rr
}



pub fn inner_product<F,G,T1,T2,T3,const R1: usize, const R2: usize>(f: F, g: G)->impl Fn(Ndarr<T1,R1>,Ndarr<T2,R2>)->Ndarr<T3,{const_max(R1 + R2 - 1, R1 + R2 - 1)-1}>
    where 
    T1: Copy + Clone + Debug + Default,
    T2: Copy + Clone + Debug + Default,
    T3: Copy + Clone + Debug + Default,
    [usize; const_max(R1, R1 + R2 - 1)]: Sized,
    [usize; const_max(R2, R2 + R1 - 1)]: Sized, //BUG: Actually a bug with rust compiler that doesn't idintyfy permutations of arithmetic operation.
    [usize; const_max(R2, R1 + R2 - 1)]: Sized,
    [usize; const_max(R1 + R2 - 1, R1 + R2 - 1)]: Sized, //same here
    [usize; const_max(R1 + R2 - 1, R2 + R1 - 1)]: Sized,
    [usize; const_max(R1 + R2 - 1, R1 + R2 - 1)-1]: Sized, //same here
    [usize; const_max(R1, R2)]: Sized,
    F: Fn(T1,T2)->T3 + Clone,
    G: Fn(T3,T3)->T3 + Clone,


{

    let out = move |arr1, arr2| {inner(f.clone(), g.clone(), arr1, arr2)};
    return out
}

pub fn outer<F,T1,T2,T3,const R1: usize, const R2: usize>(f: F, arr1: Ndarr<T1,R1>, arr2: Ndarr<T2,R2>)->Ndarr<T3,{const_max(R1 + R2, R1 + R2)}>
    where 
    T1: Copy + Clone + Debug + Default,
    T2: Copy + Clone + Debug + Default,
    T3: Copy + Clone + Debug + Default,
    [usize; const_max(R1, R1 + R2 )]: Sized,
    [usize; const_max(R2, R2 + R1)]: Sized, //BUG: Actually a bug with rust compiler that doesn't idintyfy permutations of arithmetic operation.
    [usize; const_max(R2, R1 + R2)]: Sized,
    [usize; const_max(R1 + R2, R1 + R2)]: Sized, //same here
    [usize; const_max(R1 + R2, R2 + R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
    F: Fn(T1,T2)->T3,


{
    let arr1 = arr1.t();
    let padded1: [usize; R1 + R2] = helpers::path_shape(&arr1.shape).unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr{data: bdata, shape: padded1}.t();
    let padded2: [usize; R1+ R2] = helpers::path_shape(&arr2.shape).unwrap();
    let bdata2 = arr2.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr{data: bdata2, shape: padded2};
    let r = poly_diatic(arr1, arr2, f).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
        //seems to work for all test I did
    return r
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
