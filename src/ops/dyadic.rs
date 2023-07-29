use super::*;
use rayon::{prelude::*, result};
use std::any::TypeId;
use std::cmp::{max, min};
use std::sync::{Mutex, Arc};
use std::{ops::*};
use typenum::{Max, Maximum, Sub1, Sum, Unsigned, B1};
use faer_core::{Mat,mul};

const TILE_SIZE: usize = 256;

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
        if self.shape() == other.shape() {
            let new_data: Vec<T3> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| f(a.clone(), b.clone()))
                .collect();
            return Ok(Ndarr {
                data: new_data,
                dim: Dim::<Maximum<R1, R2>>::new(&self.shape()).unwrap(),
            });
        } else {
            let new_shape = self.dim.broadcast_shape_notyped(&other.dim)?;
            let cast1 = self.broadcast_data(&other.dim)?;
            let cast2 = other.broadcast_data(&self.dim)?;

            let new_data: Vec<T3> = cast1
                .iter()
                .zip(cast2.iter())
                .map(|(a, b)| f(a.clone(), b.clone()))
                .collect();
            return Ok(Ndarr {
                data: new_data,
                dim: Dim::<Maximum<R1, R2>>::new(&new_shape.shape).unwrap(),
            });
        }
    }

    fn matmul_faer_f32(data1: &[T1], data2: &[T1], shape1: &[usize], shape2: &[usize]) -> Vec<T1>
    where
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign + 'static,
    {
        //Transforms types
        // SAFETY: 
            // 1. `T1` is `f32`: Is assumes that this function is only reachable if `T1` is `f32`.
            // 2. Memory Compatibility: Assumes compatible memory layout between `T` and `f32`.
            // 3. Drop Behavior: Transmutation bypasses `T1`'s drop behavior.
        let mut v_1 = std::mem::ManuallyDrop::new(data1.to_vec());
        let typed_1: Vec<f32> = unsafe {Vec::from_raw_parts(v_1.as_mut_ptr() as *mut f32,v_1.len(),v_1.capacity())};

        let mut v_2 = std::mem::ManuallyDrop::new(data2.to_vec());
        let typed_2: Vec<f32> = unsafe {Vec::from_raw_parts(v_2.as_mut_ptr() as *mut f32,v_2.len(),v_2.capacity())};

        let m1 = Mat::with_dims(shape1[0],shape1[1],|i, j| typed_1[shape1[1]*i+j]);
        let m2 = Mat::with_dims(shape2[0],shape2[1],|i, j| typed_2[shape2[1]*i+j]); 

        let mut out = Mat::zeros(shape1[0],shape2[1]);

        mul::matmul(
        out.as_mut(),
        m1.as_ref(),
        m2.as_ref(),
        None,
        1.0,
        faer_core::Parallelism::Rayon(0));
        
        // Transform types
        let combinations = (0..out.ncols()).flat_map(|i| (0..out.nrows()).map(move |j| (i, j))).map(|(i,j) | out.read(i, j)).collect::<Vec<_>>();
        let mut v_out = std::mem::ManuallyDrop::new(combinations);
        let out_typed: Vec<T1> = unsafe {Vec::from_raw_parts(v_out.as_mut_ptr() as *mut T1, v_out.len(),v_out.capacity())};

        out_typed
    }

    fn matmul_faer_f64(data1: &[T1], data2: &[T1], shape1: &[usize], shape2: &[usize]) -> Vec<T1>
    where
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign + 'static,
    {
       //Transforms types
        // SAFETY: 
            // 1. `T1` is `f64`: Is assumes that this function is only reachable if `T1` is `f64`.
            // 2. Memory Compatibility: Assumes compatible memory layout between `T` and `f64`.
            // 3. Drop Behavior: Transmutation bypasses `T1`'s drop behavior.
        let mut v_1 = std::mem::ManuallyDrop::new(data1.to_vec());
        let typed_1: Vec<f64> = unsafe {Vec::from_raw_parts(v_1.as_mut_ptr() as *mut f64,v_1.len(),v_1.capacity())};

        let mut v_2 = std::mem::ManuallyDrop::new(data2.to_vec());
        let typed_2: Vec<f64> = unsafe {Vec::from_raw_parts(v_2.as_mut_ptr() as *mut f64,v_2.len(),v_2.capacity())};

        let m1 = Mat::with_dims(shape1[0],shape1[1],|i, j| typed_1[shape1[1]*i+j]);
        let m2 = Mat::with_dims(shape2[0],shape2[1],|i, j| typed_2[shape2[1]*i+j]); 

        let mut out = Mat::zeros(shape1[0],shape2[1]);

        mul::matmul(
        out.as_mut(),
        m1.as_ref(),
        m2.as_ref(),
        None,
        1.0,
        faer_core::Parallelism::Rayon(0));
        
        // Transform types
        let combinations = (0..out.ncols()).flat_map(|i| (0..out.nrows()).map(move |j| (i, j))).map(|(i,j) | out.read(i, j)).collect::<Vec<_>>();
        let mut v_out = std::mem::ManuallyDrop::new(combinations);
        let out_typed: Vec<T1> = unsafe {Vec::from_raw_parts(v_out.as_mut_ptr() as *mut T1, v_out.len(),v_out.capacity())};

        out_typed
    }
    

    fn naive_2d(data1: &[T1], data2: &[T1], rows: usize, cols: usize) -> Vec<T1>
    where
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1>,
    {
        let n = cols;
        let mut result_data = vec![data1[0].clone(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = data1[i * n].clone() * data2[j].clone();
                for k in 1..n {
                    sum = sum + data1[i * n + k].clone() * data2[k * cols + j].clone();
                }
                result_data[i * cols + j] = sum;
            }
        }
        return result_data;
    }


    fn naive_2d_parallel(data1: &[T1], data2: &[T1], rows: usize, cols: usize) -> Vec<T1>
    where
        T1: Clone + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync,
    {
        let n = cols;
        let mut result_data = vec![T1::default(); rows * cols];
        result_data.par_chunks_mut(cols).enumerate().for_each(|(i, row_chunk)| {
            for j in 0..cols {
                let mut sum = data1[i * n].clone() * data2[j].clone();
                for k in 1..n {
                    sum = sum + data1[i * n + k].clone() * data2[k * cols + j].clone();
                }
                row_chunk[j] = sum;
            }
        });
        result_data
    }


    fn matmul_2d(data1: &[T1], data2: &[T1], shape1: &[usize], shape2: &[usize]) -> Vec<T1>
    where
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign + 'static,
    {
        
        if TypeId::of::<T1>() == TypeId::of::<f64>(){
            return Self::matmul_faer_f64(data1, data2, shape1, shape2);
        }else if TypeId::of::<T1>() == TypeId::of::<f32>(){
            return Self::matmul_faer_f32(data1, data2, shape1, shape2);
        }else{
            if data1.len() < 1025 && data2.len() < 1025{
                return Self::naive_2d(data1, data2, shape1[0], shape2[1])
            }else{
                return Self::naive_2d_parallel(data1, data2, shape1[0], shape2[1])
            }      
        }
    }

    pub fn mat_mul<R2: Unsigned>(&self, other: &Ndarr<T1, R2>) -> Ndarr<T1, Sub1<Sub1<Sum<R1, R2>>>>
    where
        //TODO: remove poly dyadic trait req
        R1: Max<R2>,
        <R1 as Max<R2>>::Output: Unsigned,
        R1: Add<R2>,
        <R1 as Add<R2>>::Output: Sub<B1>,
        <<R1 as Add<R2>>::Output as Sub<B1>>::Output: Sub<B1>,
        <<<R1 as Add<R2>>::Output as Sub<B1>>::Output as Sub<B1>>::Output: Unsigned,
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign + 'static,
    {
        if self.dim.len() == 2 && other.dim.len() == 2 {
            Ndarr {
                data: Self::matmul_2d(&self.data, &other.data, self.shape(), other.shape()),
                dim: Dim::<Sub1<Sub1<Sum<R1, R2>>>>::new(&[self.shape()[0], other.shape()[1]])
                    .unwrap(),
            }
        } else {
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
            Ndarr {
                data: rr.data,
                dim: Dim::<Sub1<Sub1<Sum<R1, R2>>>>::new(&rr.dim.shape).unwrap(),
            }
        }
    }

    pub fn inner_product<F, G, T2, T3, R2: Unsigned>(
        &self,
        other: &Ndarr<T2, R2>,
        f: F,
        g: G,
    ) -> Ndarr<T3, Sub1<Sub1<Sum<R1, R2>>>>
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
        let rr = r
            .reduce_notyped(self.dim.len() - 1, |x, y| g(x, y))
            .unwrap();

        Ndarr {
            data: rr.data,
            dim: Dim::<Sub1<Sub1<Sum<R1, R2>>>>::new(&rr.dim.shape).unwrap(),
        }
    }

    pub fn outer_product<F, T2, T3, R2: Unsigned>(
        &self,
        other: &Ndarr<T2, R2>,
        f: F,
    ) -> Ndarr<T3, Sum<R1, R2>>
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

        Ndarr {
            data: r.data,
            dim: Dim::<Sum<R1, R2>>::new(&r.dim.shape).unwrap(),
        }
    }
}

#[cfg(test)]

mod dyadic_test {
    use std::result;

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
        let r1 = z.outer_product(&z, |x, y| x + y);
        let r2 = z.outer_product(&z, g);
        assert_eq!(r1, Ndarr::from([[2, 3, 4], [3, 4, 5], [4, 5, 6]]));
        assert_eq!(r2, Ndarr::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
    }

    #[test]
    fn mat_mul() {
        //Test integer matmul
        let a = Ndarr::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let b = Ndarr::from([[9, 10, 11], [12, 13, 14], [15, 16, 17]]);
        let result_a_b = Ndarr::from([[42, 45, 48], [150, 162, 174], [258, 279, 300]]);

        assert_eq!(a.mat_mul(&b), result_a_b);
        //Test f64 matmul
        let a2: Ndarr<f64, U2> = Ndarr::linspace(0.1, 0.9, 9).reshape([3, 3]).unwrap();
        let b2: Ndarr<f64, U2> = Ndarr::linspace(1.1, 1.9, 9).reshape([3, 3]).unwrap();
        let result_a2_b2 = Ndarr::from([[0.9, 0.96, 1.02], [2.16, 2.31, 2.46], [3.42, 3.66, 3.9]]);
        let r = a2.mat_mul(&b2);
        assert!(r.approx(&result_a2_b2));

        //Test big f64 matmul
        let big: Ndarr<f64, U2> = Ndarr::linspace(0., 625., 62500).reshape([250, 250]).unwrap();
        let r_big = big.mat_mul(&big.t());
        assert_eq!(*r_big.index([5,5]),47362.97810317607);


        let c = Ndarr::from(0..5);
        assert_eq!(c.mat_mul(&c).scalar(), 30)
    }

    #[test]
    fn inner_product() {
        let a = Ndarr::from(1..13).reshape([2, 2, 3]).unwrap();
        let b = Ndarr::from(1..13).reshape([3, 2, 2]).unwrap();
        assert_eq!(
            a.inner_product(&b, |x, y| x * y, |x, y| x + y),
            a.mat_mul(&b)
        )
    }

}
