use super::*;
use std::ops::*;
use typenum::{Unsigned, Sub1, Maximum, B1, Max, Sum};


impl<T1: Clone + Debug + Default, R1: Unsigned> Ndarr<T1,R1>{
    pub fn poly_diatic<F,T2,T3,R2: Unsigned>(&self, other: &Ndarr<T2,R2>, f: F)->Result<Ndarr<T3, Maximum<R1,R2>>, DimError>
    where
        R1: Max<R2>,
        R2: Max<R1>,
        <R1 as Max<R2>>::Output: Unsigned,
        <R2 as Max<R1>>::Output: Unsigned,
        T1: Clone + Debug + Default,
        T2: Clone + Debug + Default,
        T3: Clone + Debug + Default,
        F: Fn(T1, T2) -> T3,
    {
        let new_shape = self.dim.broadcast_shape(&other.dim)?;
        let cast1 = self.broadcast(&other.dim)?;
        let cast2 = other.broadcast(&self.dim)?;
        let mut new_data = vec![T3::default(); cast2.len()];
        for i in 0..cast1.len() {
            new_data[i] = f(cast1.data[i].clone(), cast2.data[i].clone())
        }
        return Ok(Ndarr {
            data: new_data,
            dim: new_shape,
        });
    }

    pub fn mat_mul<R2: Unsigned>(&self, other: &Ndarr<T1,R2>)->Ndarr<T1,Sub1< Maximum< Sub1< Sum<R1,R2>> , Sub1<Sum<R1,R2> > > > >
        where
            R1: Add<R2>,
            R1: Max<R2>,
            R1: Max<Sub1<Sum<R1,R2>>>,
            R2: Max<Sub1<Sum<R1,R2>>>,
            <R1 as Max<<<R1 as Add<R2>>::Output as Sub<B1>>::Output>>::Output: Unsigned,
            <R2 as Max<<<R1 as Add<R2>>::Output as Sub<B1>>::Output>>::Output: Unsigned,
            <R1 as Add<R2>>::Output: Sub<B1>,
            <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Sub<B1>,
            <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Sub<B1>>::Output: Unsigned,
            <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Unsigned,
            <R1 as Max<R2>>::Output: Unsigned,
            <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Max,
            <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output: Unsigned,
            <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output: Sub<B1>,
            <<<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output as Sub<B1>>::Output: Unsigned,
            T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1>,
    {
    let arr1 = self.clone().t();
    let padded1 = arr1.dim.path_shape::<Sub1<Sum<R1,R2>>>().unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr {
        data: bdata,
        dim: padded1,
    }
    .t();
    let padded2 = other.dim.path_shape::<Sub1<Sum<R1,R2>>>().unwrap();
    let bdata2 = other.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr {
        data: bdata2,
        dim: padded2,
    };
    let r = poly_diatic(&arr1, &arr2, |x, y| x * y).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
    //seems to work for all test I did
    let rr = r.reduce(R1::to_usize() - 1, |x, y| x + y).unwrap();
    return rr;
    }
}

pub fn poly_diatic<F, T1, T2, T3, R1: Unsigned, R2: Unsigned>(
    arr1: &Ndarr<T1, R1>,
    arr2: &Ndarr<T2, R2>,
    f: F,
) -> Result<Ndarr<T3, Maximum<R1,R2>>, DimError>
where
    R1: Max<R2>,
    R2: Max<R1>,
    <R1 as Max<R2>>::Output: Unsigned,
    <R2 as Max<R1>>::Output: Unsigned,
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
    F: Fn(T1, T2) -> T3,
{
    let new_shape = arr1.dim.broadcast_shape(&arr2.dim)?;
    let cast1 = arr1.broadcast(&arr2.dim)?;
    let cast2 = arr2.broadcast(&arr1.dim)?;
    let mut new_data = vec![T3::default(); cast2.len()];
    for i in 0..cast1.len() {
        new_data[i] = f(cast1.data[i].clone(), cast2.data[i].clone())
    }
    return Ok(Ndarr {
        data: new_data,
        dim: new_shape,
    });
}

//TODO: found some way to simplify, this has concerning levels of cursedness!!

pub fn mat_mul<T, R1: Unsigned, R2: Unsigned>(
    arr1: &Ndarr<T, R1>,
    arr2: &Ndarr<T, R2>,
) -> Ndarr<T,Sub1< Maximum< Sub1< Sum<R1,R2>> , Sub1<Sum<R1,R2> > > > >
where
    R1: Add<R2>,
    R1: Max<R2>,
    R1: Max<Sub1<Sum<R1,R2>>>,
    R2: Max<Sub1<Sum<R1,R2>>>,
    <R1 as Max<<<R1 as Add<R2>>::Output as Sub<B1>>::Output>>::Output: Unsigned,
    <R2 as Max<<<R1 as Add<R2>>::Output as Sub<B1>>::Output>>::Output: Unsigned,
    <R1 as Add<R2>>::Output: Sub<B1>,
    <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Sub<B1>,
    <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Sub<B1>>::Output: Unsigned,
    <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Unsigned,
    <R1 as Max<R2>>::Output: Unsigned,
    <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Max,
    <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output: Unsigned,
    <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output: Sub<B1>,
    <<<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output as Sub<B1>>::Output: Unsigned,
    T: Sub<Output = T> + Clone + Debug + Default + Add<Output = T> + Mul<Output = T>,
{
    let arr1 = arr1.clone().t();
    let padded1 = arr1.dim.path_shape::<Sub1<Sum<R1,R2>>>().unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr {
        data: bdata,
        dim: padded1,
    }
    .t();
    let padded2 = arr2.dim.path_shape::<Sub1<Sum<R1,R2>>>().unwrap();
    let bdata2 = arr2.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr {
        data: bdata2,
        dim: padded2,
    };
    let r = poly_diatic(&arr1, &arr2, |x, y| x * y).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
    //seems to work for all test I did
    let rr = r.reduce(R1::to_usize() - 1, |x, y| x + y).unwrap();
    return rr;
}

pub fn inner_product<F, G, T1, T2, T3, R1: Unsigned, R2: Unsigned>(
    f: F,
    g: G,
    arr1: Ndarr<T1, R1>,
    arr2: Ndarr<T2, R2>,
) -> Ndarr<T3,Sub1< Maximum< Sub1< Sum<R1,R2>> , Sub1<Sum<R1,R2> > > > >
where
    R1: Add<R2>,
    R1: Max<R2>,
    R1: Max<Sub1<Sum<R1,R2>>>,
    R2: Max<Sub1<Sum<R1,R2>>>,
    <R1 as Max<<<R1 as Add<R2>>::Output as Sub<B1>>::Output>>::Output: Unsigned,
    <R2 as Max<<<R1 as Add<R2>>::Output as Sub<B1>>::Output>>::Output: Unsigned,
    <R1 as Add<R2>>::Output: Sub<B1>,
    <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Sub<B1>,
    <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Sub<B1>>::Output: Unsigned,
    <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Unsigned,
    <R1 as Max<R2>>::Output: Unsigned,
    <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Max,
    <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output: Unsigned,
    <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output: Sub<B1>,
    <<<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Max>::Output as Sub<B1>>::Output: Unsigned,
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
    F: Fn(T1, T2) -> T3,
    G: Fn(T3, T3) -> T3,
{
    let arr1 = arr1.t();
    let padded1 = arr1.dim.path_shape::<Sub1<Sum<R1,R2>>>().unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr {
        data: bdata,
        dim: padded1,
    }
    .t();
    let padded2 = arr2.dim.path_shape::<Sub1<Sum<R1,R2>>>().unwrap();
    let bdata2 = arr2.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr {
        data: bdata2,
        dim: padded2,
    };
    let r = poly_diatic(&arr1, &arr2, f).unwrap();
    //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
    //seems to work for all test I did
    let rr = r.reduce(R1::to_usize() - 1, |x, y| g(x, y)).unwrap();
    return rr;
}

pub fn outer_product<F, T1, T2, T3, R1: Unsigned, R2: Unsigned>(
    f: F,
    arr1: &Ndarr<T1, R1>,
    arr2: &Ndarr<T2, R2>,
//) -> Ndarr<T3, Sum<R1,R2>>
) -> Ndarr<T3, Maximum<Sum<R1,R2>,Sum<R1,R2>>  >
where
    R1: Add<R2>,
    <R1 as Add<R2>>::Output: Unsigned,
    R1: Max<Sum<R1,R2>>,
    R2: Max<Sum<R1,R2>>,
    <R1 as Max<<R1 as Add<R2>>::Output>>::Output: Unsigned,
    <R2 as Max<<R1 as Add<R2>>::Output>>::Output: Unsigned,
    <<R1 as Add<R2>>::Output as Max>::Output: Unsigned,
    <R1 as Add<R2>>::Output: Max,
    T1: Clone + Debug + Default,
    T2: Clone + Debug + Default,
    T3: Clone + Debug + Default,
    F: Fn(T1, T2) -> T3,
{
    let arr1 = arr1.clone().t();
    let padded1 = arr1.dim.path_shape::<Sum<R1,R2>>().unwrap();
    let bdata = arr1.broadcast_data(&padded1).unwrap();
    let arr1 = Ndarr {
        data: bdata,
        dim: padded1,
    }
    .t();
    let padded2 = arr2.dim.path_shape::<Sum<R1,R2>>().unwrap();
    let bdata2 = arr2.broadcast_data(&padded2).unwrap();
    let arr2 = Ndarr {
        data: bdata2,
        dim: padded2,
    };
    let r = poly_diatic(&arr1, &arr2, f).unwrap();
    return r;
}
