use crate::{scalars::{Scalar, Float}, helpers::{const_max}};

use super::*;
use std::ops::*;


pub fn poly_diatic<F,T1,T2,T3, const R1: usize, const R2: usize>(arr1: Ndarr<T1,R1>, arr2: Ndarr<T2,R2>, f: F)->Result<Ndarr<T3,{const_max(R1,  R2)}>,DimError>
where
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
    F: Fn(T1,T2) -> T3,
    [usize; const_max(R2, R1)]: Sized,
{
    let new_shape = helpers::broadcast_shape(&arr1.shape, &arr2.shape)?;
    let cast1 = arr1.broadcast(&arr2.shape)?; 
    let cast2 = arr2.broadcast(&arr1.shape)?;
    let mut new_data = vec![T3::default(); cast2.len()];
    for i in 0..cast1.len(){
        new_data[i] = f(cast1.data[i].clone(), cast2.data[i].clone())
    }
    return Ok(Ndarr { data: new_data , shape: new_shape });
}


//TODO: found some way to simplify, this has concerning levels of cursedness!!

pub fn mat_mul<T,const R1: usize, const R2: usize>(arr1: Ndarr<T,R1>, arr2: Ndarr<T,R2>)->Ndarr<T,{const_max(R1 + R2 - 1, R1 + R2 - 1)-1}>
    where T: Sub<Output = T> + Clone + Debug + Default + Add<Output = T> + Mul<Output = T>, 
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
    let rr = r.reduce(R1-1, |x,y| x+y).unwrap();
    return rr
}




pub fn inner_product<F,G,T1,T2,T3,const R1: usize, const R2: usize>(f: F, g: G, arr1: Ndarr<T1,R1>, arr2: Ndarr<T2,R2>)->Ndarr<T3,{const_max(R1 + R2 - 1, R1 + R2 - 1)-1}>
    where 
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
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
    let rr = r.reduce(R1-1, |x,y| g(x,y)).unwrap();
    return rr
}



pub fn inner_closure<F,G,T1,T2,T3,const R1: usize, const R2: usize>(f: F, g: G)->impl Fn(Ndarr<T1,R1>,Ndarr<T2,R2>)->Ndarr<T3,{const_max(R1 + R2 - 1, R1 + R2 - 1)-1}>
    where 
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
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

    let out = move |arr1, arr2| {inner_product(f.clone(), g.clone(), arr1, arr2)};
    return out
}

pub fn outer_product<F,T1,T2,T3,const R1: usize, const R2: usize>(f: F, arr1: Ndarr<T1,R1>, arr2: Ndarr<T2,R2>)->Ndarr<T3,{const_max(R1 + R2, R1 + R2)}>
    where 
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
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



// Arithmetic operations
//TODO: a lot of code repetition, need to refactor this with a macro. Also there should be a better way to handel the permutations of
    //owned and reference. But if we want the prioritize flexibility and friendliness we really need to handel this permutations.
//////////////////////////////////////////// Add /////////////////////////////////////////////

impl <T1, const R1: usize, const R2: usize> Add<Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Add<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn add(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs, |x,y| x + y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Add<&Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Add<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn add(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs.clone(), |x,y| x + y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Add<Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Add<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn add(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x + y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Add<&Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Add<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn add(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x + y).unwrap()
    }
}

impl<P, T, const R: usize> Add<P> for Ndarr<T, R>
where
    T: Add<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Self;
    fn add(self, other: P) -> Self::Output {
        let other = other.into_ndarr(&self.shape);
        self.bimap(&other, |x, y| x + y)
    }
}

impl<P, T, const R: usize> Add<P> for &Ndarr<T, R>
where
    T: Add<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Ndarr<T, R>;
    fn add(self, other: P) -> Self::Output {
        let other = other.into_ndarr(&self.shape);
        self.clone().bimap(&other, |x, y| x + y)
    }
}


//////////////////////////////////////////// Sub /////////////////////////////////////////////

impl <T1, const R1: usize, const R2: usize> Sub<Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Sub<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn sub(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs, |x,y| x - y).unwrap()
    }
}


impl <T1, const R1: usize, const R2: usize> Sub<&Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Sub<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn sub(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs.clone(), |x,y| x - y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Sub<&Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Sub<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn sub(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x - y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Sub<Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Sub<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn sub(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x - y).unwrap()
    }
}

impl<P, T, const R: usize> Sub<P> for Ndarr<T, R>
where
    T: Sub<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Self;
    fn sub(self, other: P) -> Self::Output {
        let other = other.into_ndarr(&self.shape);
        self.bimap(&other, |x, y| x - y)
    }
}

impl<P, T, const R: usize> Sub<P> for &Ndarr<T, R>
where
    T: Sub<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Ndarr<T, R>;
    fn sub(self, other: P) -> Self::Output {
        let other = other.into_ndarr(&self.shape);
        self.clone().bimap(&other, |x, y| x - y)
    }
}



//////////////////////////////////////////// Mul /////////////////////////////////////////////

impl <T1, const R1: usize, const R2: usize> Mul<Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Mul<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn mul(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs, |x,y| x * y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Mul<&Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Mul<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn mul(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs.clone(), |x,y| x * y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Mul<&Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Mul<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn mul(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x * y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Mul<Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Mul<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn mul(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs, |x,y| x * y).unwrap()
    }
}

impl<P, T, const R: usize> Mul<P> for Ndarr<T, R>
where
    T: Mul<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Self;
    fn mul(self, other: P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.bimap(&other, |x, y| x * y)
    }
}

impl<P, T, const R: usize> Mul<P> for &Ndarr<T, R>
where
    T: Mul<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Ndarr<T, R>;
    fn mul(self, other: P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.clone().bimap(&other, |x, y| x * y)
    }
}



//////////////////////////////////////////// Div /////////////////////////////////////////////

impl <T1, const R1: usize, const R2: usize> Div<Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Div<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn div(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs, |x,y| x / y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Div<&Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Div<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn div(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs.clone(), |x,y| x / y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Div<&Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Div<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn div(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x / y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Div<Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Div<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn div(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs, |x,y| x / y).unwrap()
    }
}

impl<P, T, const R: usize> Div<P> for Ndarr<T, R>
where
    T: Div<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Self;
    fn div(self, other: P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.bimap(&other, |x, y| x / y)
    }
}

impl<P, T, const R: usize> Div<P> for &Ndarr<T, R>
where
    T: Div<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Ndarr<T, R>;
    fn div(self, other: P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.clone().bimap(&other, |x, y| x / y)
    }
}


//////////////////////////////////////////// Rem /////////////////////////////////////////////

impl <T1, const R1: usize, const R2: usize> Rem<Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Rem<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn rem(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs, |x,y| x % y).unwrap()
    }
}


impl <T1, const R1: usize, const R2: usize> Rem<&Ndarr<T1,R2>> for Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Rem<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn rem(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self, rhs.clone(), |x,y| x % y).unwrap()
    }
}


impl <T1, const R1: usize, const R2: usize> Rem<&Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Rem<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn rem(self, rhs: &Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs.clone(), |x,y| x % y).unwrap()
    }
}

impl <T1, const R1: usize, const R2: usize> Rem<Ndarr<T1,R2>> for &Ndarr<T1,R1>
where
    T1: Clone + Debug + Default + Rem<Output = T1>,
    [usize; const_max(R2, R1)]: Sized,
    [usize; const_max(R1, R2)]: Sized,
{
    type Output = Ndarr<T1,{const_max(R1,  R2)}>;
    fn rem(self, rhs: Ndarr<T1,R2>) -> Self::Output {
        poly_diatic(self.clone(), rhs, |x,y| x % y).unwrap()
    }
}

impl<P, T, const R: usize> Rem<P> for Ndarr<T, R>
where
    T: Rem<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Self;
    fn rem(self, other: P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.bimap(&other, |x, y| x % y)
    }
}

impl<P, T, const R: usize> Rem<P> for &Ndarr<T, R>
where
    T: Rem<Output = T> +  Clone + Debug + Default,
    P: IntoNdarr<T, R> + Scalar,
{
    type Output = Ndarr<T, R>;
    fn rem(self, other: P) -> Self::Output {
        //this is temporary, util we att projection por rank polymorphic operations
        let other = other.into_ndarr(&self.shape);
        self.clone().bimap(&other, |x, y| x % y)
    }
}


//////////////////////////////////////////// Neg /////////////////////////////////////////////

impl<T, const R: usize> Neg for Ndarr<T, R>
where
    T: Neg<Output = T> +  Clone + Debug + Default + Copy,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}


impl<T, const R: usize> Neg for &Ndarr<T, R>
where
    T: Neg<Output = T> +  Clone + Debug + Default + Copy,
{
    type Output = Ndarr<T,R>;
    fn neg(self) -> Self::Output {
        self.clone().map(|x| -*x)
    }
}

//////////////////////////////////////////// AddAssing /////////////////////////////////////////////


impl<P, T, const R: usize> AddAssign<&P> for Ndarr<T, R>
where
    T: Add<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T,R> + Clone,
{
    fn add_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x,y| x+y)
    }
}


////////////////////////////////////////////  SubAssing /////////////////////////////////////////////

impl<P, T, const R: usize> SubAssign<&P> for Ndarr<T, R>
where
    T: Sub<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T,R> + Clone,
{
    fn sub_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x,y| x-y)
    }
}


////////////////////////////////////////////  MulAssing /////////////////////////////////////////////

impl<P, T, const R: usize> MulAssign<&P> for Ndarr<T, R>
where
    T: Mul<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T,R> + Clone,
{
    fn mul_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x,y| x*y)
    }
}

////////////////////////////////////////////  DivAssing /////////////////////////////////////////////

impl<P, T, const R: usize> DivAssign<&P> for Ndarr<T, R>
where
    T: Div<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T,R> + Clone,
{
    fn div_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x,y| x/y)
    }
}

////////////////////////////////////////////  RemAssing /////////////////////////////////////////////

impl<P, T, const R: usize> RemAssign<&P> for Ndarr<T, R>
where
    T: Rem<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T,R> + Clone,
{
    fn rem_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x,y| x%y)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////  Trig Functions /////////////////////////////////////////////
impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Copy + Debug + Default + Float
{
    pub fn sin(&self)->Self{
        let out = self.clone().map(|x| x.f_sin());
        out
    }

    pub fn cos(&self)->Self{
        let out = self.clone().map(|x| x.f_cos());
        out
    }

    pub fn tan(&self)->Self{
        let out = self.clone().map(|x| x.f_tan());
        out
    }

    pub fn sinh(&self)->Self{
        let out = self.clone().map(|x| x.f_sinh());
        out
    }
    
    pub fn cosh(&self)->Self{
        let out = self.clone().map(|x| x.f_cosh());
        out
    }

    pub fn tanh(&self)->Self{
        let out = self.clone().map(|x| x.f_tanh());
        out
    }

    pub fn log(&self, base: T)->Self{
        let out = self.clone().map(|x| x.f_log(base));
        out
    }
    
    pub fn ln(&self)->Self{
        let out = self.clone().map(|x| x.f_ln());
        out
    }

    pub fn log2(&self)->Self{
        let out = self.clone().map(|x| x.f_log2());
        out
    }

    
   

}

// TODO: find a better way
pub fn abs_num<T>(n: T)->T
where T: From<i8> + Mul<Output = T> + PartialOrd
{
    if n < Into::<T>::into(0) {
        return Into::<T>::into(-1) * n;
    }else{
        return n;
    }

}

impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Debug + Default + From<i8> + Mul<Output = T> + PartialOrd
{
    pub fn abs(&self)->Self{
       let out = self.clone().map(|x| abs_num(x.clone()));
       out
    }
}