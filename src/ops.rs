
use num_traits::{Float, Signed};
use crate::{scalars::{Scalar}, helpers::{const_max}};
use super::*;
use std::ops::*;


pub fn poly_diatic<F,T1,T2,T3, const R1: usize, const R2: usize>(arr1: &Ndarr<T1,R1>, arr2: &Ndarr<T2,R2>, f: F)->Result<Ndarr<T3,{const_max(R1,  R2)}>,DimError>
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
    let r = poly_diatic(&arr1, &arr2, |x,y| x*y).unwrap();
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
    let r = poly_diatic(&arr1, &arr2, f).unwrap();
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
    let r = poly_diatic(&arr1, &arr2, f).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
        //seems to work for all test I did
    return r
}



macro_rules!  ndarr_op{
    ($Ty1:ty, $Ty2:ty, $Trait:tt, $F:tt, $Op:tt) => {
        
        impl <T1, T2, T3, const R1: usize, const R2: usize> $Trait<$Ty2> for $Ty1
        where
            T1: Clone + Debug + Default + $Trait<T2, Output = T3>,
            T2: Clone + Debug + Default,
            T3: Clone + Debug + Default,
            [usize; const_max(R2, R1)]: Sized,
            [usize; const_max(R1, R2)]: Sized,
        {
            type Output = Ndarr<T3,{const_max(R1,  R2)}>;
            fn $F(self, rhs: $Ty2) -> Self::Output {
                poly_diatic(&self, &rhs, |x,y| x $Op y).unwrap()
            }
        }
    };
}
//--------------------------------- Add --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Add, add, +);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Add, add, +);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Add, add, +);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Add, add, +);

//--------------------------------- Sub --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Sub, sub, -);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Sub, sub, -);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Sub, sub, -);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Sub, sub, -);

//--------------------------------- Mul --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Mul, mul, *);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Mul, mul, *);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Mul, mul, *);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Mul, mul, *);

//--------------------------------- Div --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Div, div, /);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Div, div, /);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Div, div, /);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Div, div, /);

//--------------------------------- Rem --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Rem, rem, %);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Rem, rem, %);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Rem, rem, %);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Rem, rem, %);

//////////////////////////////// Scalars ////////////////////////////////////
macro_rules! scalar_op {
    ($Op:tt, $f_name:tt, $f:tt) => {
        impl<L,P, T, const R: usize> $Op<P> for Ndarr<T, R>
        where
            L: Clone + Debug + Default,
            T: Clone + Debug + Default + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L,R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map_types(|x| x.clone() $f other)
            }
        }
        impl<L,P, T, const R: usize> $Op<P> for &Ndarr<T, R>
        where
            L: Clone + Debug + Default,
            T: Clone + Debug + Default + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L, R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map_types(|x| x.clone() $f other)
            }
        }
    };
}

scalar_op!(Add, add, +);
scalar_op!(Sub, sub, -);
scalar_op!(Mul, mul, *);
scalar_op!(Div, div, /);
scalar_op!(Rem, rem, %);


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
        self.map(|x| -*x)
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

////////////////////////////////////////////  Float Functions /////////////////////////////////////////////
impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Copy + Debug + Default + Float
{
    pub fn sin(&self)->Self{
        let out = self.map(|x| x.sin());
        out
    }

    pub fn cos(&self)->Self{
        let out = self.map(|x| x.cos());
        out
    }

    pub fn tan(&self)->Self{
        let out = self.map(|x| x.tan());
        out
    }

    pub fn sinh(&self)->Self{
        let out = self.map(|x| x.sinh());
        out
    }
    
    pub fn cosh(&self)->Self{
        let out = self.map(|x| x.cosh());
        out
    }

    pub fn tanh(&self)->Self{
        let out = self.map(|x| x.tanh());
        out
    }

    pub fn log(&self, base: T)->Self{
        let out = self.map(|x| x.log(base));
        out
    }
    
    pub fn ln(&self)->Self{
        let out = self.map(|x| x.ln());
        out
    }

    pub fn log2(&self)->Self{
        let out = self.map(|x| x.log2());
        out
    }
    pub fn is_infinite(&self) -> Ndarr<bool,R> {
       let out = self.map_types(|x| x.is_infinite() );
       out
    }
    pub fn is_finite(&self) -> Ndarr<bool,R> {
       let out = self.map_types(|x| x.is_finite() );
       out
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
       let out = self.map_types(|x| x.is_normal() );
       out

    }
    pub fn is_nan(&self) -> Ndarr<bool,R>{
       let out = self.map_types(|x| x.is_nan() );
       out
    }
}

//---------------------- Signed ---------------------
impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Debug + Default + Signed + PartialOrd
{
    pub fn abs(&self)->Self{
       let out = self.map(|x| x.abs());
       out
    }
    pub fn is_positive(&self)->Ndarr<bool, R>{
       let out = self.map_types(|x| x.is_positive());
       out
    }

    pub fn is_negative(&self)->Ndarr<bool, R>{
       let out = self.map_types(|x| x.is_negative());
       out
    }
}

impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Debug + Default + Add<Output = T> + PartialOrd
{
    pub fn sum(&self)->T{
        let data = self.data.clone();
        let mut sum = data[0].clone();
        for t in data{
            sum = sum + t;
        }
        return sum;
    }
}