use super::*;
use rayon::prelude::*;
use std::cmp::{max, min};
use std::sync::Mutex;
use std::{ops::*, result};
use typenum::{Max, Maximum, Sub1, Sum, Unsigned, B1};

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

    fn matmul_2d(&self, data1: &[T1], data2: &[T1], shape1: &[usize], shape2: &[usize]) -> Vec<T1>
    where
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign,
    {
        if data1.len() < 1025 && data2.len() < 1025 {
            return self.naive_2d(data1, data2, shape1[0], shape2[1]);
        } else {
            return self.paralell_2d(data1, data2, shape1, shape2);
        }
    }
    fn naive_2d(&self, data1: &[T1], data2: &[T1], rows: usize, cols: usize) -> Vec<T1>
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

    fn paralell_2d(&self, data1: &[T1], data2: &[T1], shape1: &[usize], shape2: &[usize]) -> Vec<T1>
    where
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign,
    {
        let inners = shape1[1];
        let rows = shape1[0];
        let columns = shape2[1];

        let num_row_tiles = (rows + TILE_SIZE - 1) / TILE_SIZE;
        let num_column_tiles = (columns + TILE_SIZE - 1) / TILE_SIZE;
        let num_inner_tiles = (inners + TILE_SIZE - 1) / TILE_SIZE;

        let result: Vec<T1> = vec![T1::default(); rows * columns];
        let result_mutex = Mutex::new(result);

        (0..num_inner_tiles).into_par_iter().for_each(|inner_tile| {
            let inner_start = inner_tile * TILE_SIZE;
            let inner_end = min(inners, inner_start + TILE_SIZE);

            for row_tile in 0..num_row_tiles {
                let row_start = row_tile * TILE_SIZE;
                let row_end = min(rows, row_start + TILE_SIZE);

                for column_tile in 0..num_column_tiles {
                    let column_start = column_tile * TILE_SIZE;
                    let column_end = min(columns, column_start + TILE_SIZE);

                    for tinner in inner_start..inner_end {
                        for trow in row_start..row_end {
                            let mut temp = data1[trow * inners + tinner].clone()
                                * data2[tinner * columns].clone();

                            for tcol in (column_start + 1)..column_end {
                                temp += data1[trow * inners + tinner].clone()
                                    * data2[tinner * columns + tcol].clone();
                            }
                            let mut result_lock = result_mutex.lock().unwrap();
                            result_lock[trow * columns + column_start + inner_tile] += temp;
                        }
                    }
                }
            }
        });
        result_mutex.into_inner().unwrap()
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
        T1: Clone + Debug + Default + Add<Output = T1> + Mul<Output = T1> + Send + Sync + AddAssign,
    {
        if R1::to_usize() == 2 && R2::to_usize() == 2 {
            Ndarr {
                data: self.matmul_2d(&self.data, &other.data, self.shape(), other.shape()),
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
        let a = Ndarr::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let b = Ndarr::from([[9, 10, 11], [12, 13, 14], [15, 16, 17]]);
        let result_a_b = Ndarr::from([[42, 45, 48], [150, 162, 174], [258, 279, 300]]);

        assert_eq!(a.mat_mul(&b), result_a_b);

        let a2: Ndarr<f64, U2> = Ndarr::linspace(0.1, 0.9, 9).reshape([3, 3]).unwrap();
        let b2: Ndarr<f64, U2> = Ndarr::linspace(1.1, 1.9, 9).reshape([3, 3]).unwrap();
        let result_a2_b2 = Ndarr::from([[0.9, 0.96, 1.02], [2.16, 2.31, 2.46], [3.42, 3.66, 3.9]]);
        assert!(a2.mat_mul(&b2).approx(&result_a2_b2));
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
