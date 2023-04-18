
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
