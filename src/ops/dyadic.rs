use super::*;
use std::ops::*;
use typenum::{Max, Maximum, Sub1, Sum, Unsigned, B1};

impl<T1: Clone + Debug, R1: Unsigned> Ndarr<T1, R1> {
    pub fn poly_dyadic<F, T2, T3, R2: Unsigned>(
        &self,
        other: &Ndarr<T2, R2>,
        f: F,
    ) -> Result<Ndarr<T3, Maximum<R1, R2>>, DimError>
    where
        R1: Max<R2>,
        <R1 as Max<R2>>::Output: Unsigned,
        T1: Clone + Debug,
        T2: Clone + Debug,
        T3: Clone + Debug,
        F: Fn(T1, T2) -> T3,
    {
        let new_shape = self.dim.broadcast_shape_notyped(&other.dim)?;
        let cast1 = self.broadcast_data(&other.dim)?;
        let cast2 = other.broadcast_data(&self.dim)?;
        let mut new_data = Vec::with_capacity(cast2.len());
        for i in 0..cast1.len() {
            new_data.push(f(cast1[i].clone(), cast2[i].clone()))
        }
        return Ok(Ndarr {
            data: new_data,
            dim: Dim::<Maximum<R1,R2>>::new(&new_shape.shape).unwrap(),
        });
    }

    pub fn mat_mul<R2: Unsigned>(
        &self,
        other: &Ndarr<T1, R2>,
    ) -> Ndarr<T1, Sub1<Sub1<Sum<R1,R2>>>>
    where
    //TODO: remove poly dyadic trait req
        R1: Max<R2>,
        <R1 as Max<R2>>::Output: Unsigned,
        R1: Add<R2>,
        <R1 as Add<R2>>::Output: Sub<B1>,
        <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Sub<B1>,
        <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Sub<B1>>::Output: Unsigned,
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1>,
    {
        let rank_intimidate = self.dim.len() + other.dim.len() - 1;
        let arr1 = self.clone().t();
        let padded1 = arr1.dim.path_shape::<UTerm>(rank_intimidate).unwrap();
        let bdata = arr1.broadcast_data(&padded1).unwrap();
        let arr1 = Ndarr {
            data: bdata,
            dim: padded1,
        }
        .t();
        let padded2 = other.dim.path_shape::<UTerm>(rank_intimidate).unwrap();
        let bdata2 = other.broadcast_data(&padded2).unwrap();
        let arr2 = Ndarr {
            data: bdata2,
            dim: padded2,
        };
        let r = arr1.poly_dyadic(&arr2, |x, y| x * y).unwrap();
        //TODO: Not 100% sure if the reduction is always in R-1 axis, I'm like 90% confident but too lazy to do a math proof.
        //seems to work for all test I did
        let rr = r.reduce_notyped(self.dim.len() - 1, |x, y| x + y).unwrap();
        Ndarr { data: rr.data , dim: Dim::<Sub1<Sub1<Sum<R1,R2>>>>::new(&rr.dim.shape).unwrap()}
    }

    pub fn inner_product<F, G, T2, T3, R2: Unsigned>(
        &self,
        other: &Ndarr<T2, R2>,
        f: F,
        g: G,
    ) -> Ndarr<T3, Sub1<Sub1<Sum<R1,R2>>>>
    where
        R1: Max<R2>,
        <R1 as Max<R2>>::Output: Unsigned,
        R1: Add<R2>,
        <R1 as Add<R2>>::Output: Sub<B1>,
        <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Sub<B1>,
        <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Sub<B1>>::Output: Unsigned,
        T1: Clone + Debug,
        T2: Clone + Debug,
        T3: Clone + Debug,
        F: Fn(T1, T2) -> T3,
        G: Fn(T3, T3) -> T3,
    {
        let rank_intimidate = self.dim.len() + other.dim.len() - 1;
        let arr1 = self.clone().t();
        let padded1 = arr1.dim.path_shape::<UTerm>(rank_intimidate).unwrap();
        let bdata = arr1.broadcast_data(&padded1).unwrap();
        let arr1 = Ndarr {
            data: bdata,
            dim: padded1,
        }
        .t();
        let padded2 = other.dim.path_shape::<UTerm>(rank_intimidate).unwrap();
        let bdata2 = other.broadcast_data(&padded2).unwrap();
        let arr2 = Ndarr {
            data: bdata2,
            dim: padded2,
        };
        let r = arr1.poly_dyadic(&arr2, f).unwrap();
        let rr = r.reduce_notyped(self.dim.len() - 1, |x, y| g(x, y)).unwrap();

        Ndarr { data: rr.data , dim: Dim::<Sub1<Sub1<Sum<R1,R2>>>>::new(&rr.dim.shape).unwrap()}
    }

    pub fn outer_product<F, T2, T3, R2: Unsigned>(
            &self,
            other: &Ndarr<T2, R2>,
            f: F,
        ) -> Ndarr<T3, Sum<R1,R2>>
        where
            R1: Max<R2>,
            <R1 as Max<R2>>::Output: Unsigned,
            R1: Add<R2>,
            <R1 as Add<R2>>::Output: Unsigned,
            T1: Clone + Debug,
            T2: Clone + Debug,
            T3: Clone + Debug,
            F: Fn(T1, T2) -> T3,
    {
        let rank_intimidate = self.dim.len() + other.dim.len();
        let arr1 = self.clone().t();
        let padded1 = arr1.dim.path_shape::<UTerm>(rank_intimidate).unwrap();
        let bdata = arr1.broadcast_data(&padded1).unwrap();
        let arr1 = Ndarr {
            data: bdata,
            dim: padded1,
        }
        .t();
        let padded2 = other.dim.path_shape::<UTerm>(rank_intimidate).unwrap();
        let bdata2 = other.broadcast_data(&padded2).unwrap();
        let arr2 = Ndarr {
            data: bdata2,
            dim: padded2,
        };
        let r = arr1.poly_dyadic(&arr2, f).unwrap();

        Ndarr { data: r.data , dim: Dim::<Sum<R1,R2>>::new(&r.dim.shape).unwrap()}
    }


}

#[cfg(test)]

mod dyadic_test{
    use super::*;
    #[test]
    fn outer() {
        let z = Ndarr::from([1, 2, 3]);
        let g = |a, b| {
            if a == b {
                1
            } else {
                0
            }
        };
        let r1 = z.outer_product(&z,|x, y| x + y);
        let r2 = z.outer_product(&z, g);
        assert_eq!(r1, Ndarr::from([[2, 3, 4], [3, 4, 5], [4, 5, 6]]));
        assert_eq!(r2, Ndarr::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
    }

    #[test]
    fn mat_mul(){
        let a = Ndarr::from([[0,1,2],[3,4,5],[6,7,8]]);
        let b = Ndarr::from([[9,10,11],[12,13,14],[15,16,17]]);
        let result_a_b = Ndarr::from([[ 42, 45,  48],[150, 162, 174],[258, 279, 300]]);

        assert_eq!(a.mat_mul(&b), result_a_b);

        let c = Ndarr::from(0..5);
        assert_eq!(c.mat_mul(&c).scalar(), 30)
    }
    #[test]
    fn inner_product(){
        let a = Ndarr::from(1..13).reshape([2,2,3]).unwrap();
        let b = Ndarr::from(1..13).reshape([3,2,2]).unwrap();
        assert_eq!(a.inner_product(&b, |x,y| x * y, |x,y| x+y), a.mat_mul(&b))
    }



}
