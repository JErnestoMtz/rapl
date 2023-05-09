//!Note: `rapl` is in early development and is  not optimized for performance, thus is not recommended for production applications.

//!`rapl` is an experimental numerical computing Rust library that provides a simple way of working with N-dimensional array, along with a wide range of mathematical functions to manipulate them. It takes inspiration from NumPy and APL, with the primary aim of achieving maximum ergonomics and user-friendliness while maintaining generality. Our goal is to make Rust scripting as productive as possible and a real option for numerical computing.
//!```
//!use rapl::*;
//!fn main() {
//!     let a = Ndarr::from([1, 2, 3]);
//!     let b = Ndarr::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
//!     let r = a + b - 1;
//!     assert_eq!(r, Ndarr::from([[1, 3, 5], [4, 6, 8], [7, 9, 11]]));
//!}
//!```

mod display;
mod errors;
mod helpers;
mod natives;
pub mod ops;
mod scalars;
mod shape;
pub mod utils;

#[cfg(feature = "complex")]
mod complex_tensor;
use std::{fmt::Debug, fmt::Display};
#[cfg(feature = "complex")]
pub mod complex;

pub use errors::DimError;
use num_traits::Float;
pub use scalars::Scalar;

#[cfg(feature = "complex")]
pub use complex::*;

pub use shape::Dim;

pub use typenum::{UTerm, B0, B1, U0, U1, U2, U3, U4, U5, U6, U7, U8};

use std::ops::{Add, Sub};
use typenum::{Add1, Max, Maximum, Sub1, Unsigned};

///Main struct of N Dimensional generic array. The shape is denoted by the `shape` array where the length is the Rank of the Ndarray the actual values are stored in a flattened state in a rank 1 array.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ndarr<T: Clone + Default, R: Unsigned> {
    pub data: Vec<T>,
    pub dim: Dim<R>,
}

impl<T: Clone + Debug + Default, R: Unsigned> Ndarr<T, R> {
    pub fn new<D: Into<Dim<R>>>(data: &[T], shape: D) -> Result<Self, DimError> {
        let shape = shape.into();
        let n = helpers::multiply_list(&shape.shape, 1);
        if data.len() == n {
            Ok(Ndarr {
                data: data.to_vec(),
                dim: shape,
            })
        } else {
            Err(DimError::new(&format!(
                "The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.",
                shape.shape,
                n,
                data.len()
            )))
        }
    }
    pub fn rank(&self) -> usize {
        self.dim.shape.len()
    }
    pub fn shape(&self) -> &[usize] {
        &self.dim.shape
    }
    pub fn flatten(self) -> Vec<T> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn from<P: Into<Self>>(p: P) -> Self {
        p.into()
    }

    pub fn reshape<R2: Unsigned, D: Into<Dim<R2>>>(
        &self,
        shape: D,
    ) -> Result<Ndarr<T, R2>, DimError> {
        let shape = shape.into();
        if helpers::multiply_list(&self.dim.shape, 1) != helpers::multiply_list(&shape.shape, 1) {
            return Err(DimError::new(&format!(
                "Can not reshape array with shape {:?} to {:?}.",
                &self.dim.shape, shape.shape
            )));
        }
        Ok(Ndarr {
            data: self.data.clone(),
            dim: shape.clone(),
        })
    }

    pub fn slice_at(&self, axis: usize) -> Vec<Ndarr<T, Sub1<R>>>
    where
        R: Sub<B1>,
        <R as Sub<B1>>::Output: Unsigned,
    {
        let n = helpers::multiply_list(&self.dim.shape, 1); // number of elements in original array
        let new_shape = self.dim.clone().remove_element(axis);
        let n_new_arrs = self.dim.shape[axis]; // number of new arrays

        let iota = 0..n;

        let indexes: Vec<Dim<R>> = iota.map(|i| self.dim.get_indexes(&i)).collect(); //indexes of each element

        let mut out = Vec::new(); // to store

        for i in 0..n_new_arrs {
            let mut this_data: Vec<T> = Vec::new();
            for j in 0..n {
                // if the index at the slice position coincide with i (or the slice number)
                if indexes[j].shape[axis] == i {
                    let ind = self.dim.get_flat_pos(&indexes[j]).unwrap();
                    this_data.push(self.data[ind].clone())
                }
            }
            //TODO: remove push, with allocation size
            out.push(Ndarr::new(&this_data, new_shape.clone()).expect("Error initializing"))
        }
        out
    }

    pub fn slice_at_notyped(&self, axis: usize) -> Vec<Ndarr<T, UTerm>> {
        let n = helpers::multiply_list(&self.dim.shape, 1); // number of elements in original array
        let new_shape = self.dim.clone().remove_element_notyped(axis);
        let n_new_arrs = self.dim.shape[axis]; // number of new arrays

        let iota = 0..n;

        let indexes: Vec<Dim<R>> = iota.map(|i| self.dim.get_indexes(&i)).collect(); //indexes of each element

        let mut out = Vec::new(); // to store

        for i in 0..n_new_arrs {
            let mut this_data: Vec<T> = Vec::new();
            for j in 0..n {
                // if the index at the slice position coincide with i (or the slice number)
                if indexes[j].shape[axis] == i {
                    let ind = self.dim.get_flat_pos(&indexes[j]).unwrap();
                    this_data.push(self.data[ind].clone())
                }
            }
            //TODO: remove push, with allocation size
            out.push(Ndarr::new(&this_data, new_shape.clone()).expect("Error initializing"))
        }
        out
    }

    pub fn reduce<F: Fn(T, T) -> T + Clone>(
        &self,
        axis: usize,
        f: F,
    ) -> Result<Ndarr<T, Sub1<R>>, DimError>
    where
        R: Sub<B1>,
        <R as Sub<B1>>::Output: Unsigned,
    {
        if axis >= R::to_usize() {
            Err(DimError::new("Axis grater than rank"))
        } else {
            let slices = self.clone().slice_at(axis);
            let n = slices.len();
            let mut out = slices[0].clone();
            for i in 1..n {
                out.bimap_in_place(&slices[i], f.clone())
            }

            Ok(out)
        }
    }
    //similar to broadcast but, this do not allow a shape different to shape
    pub fn broadcast_to<R2: Unsigned, D: Into<Dim<R2>>>(
        &self,
        shape: D,
    ) -> Result<Ndarr<T, Maximum<R, R2>>, DimError>
    where
        R: Max<R2>,
        <R as Max<R2>>::Output: Unsigned,
    {
        let shape = shape.into();
        //see https://numpy.org/doc/stable/user/basics.broadcasting.html
        //TODO: not sure at all if this implementation is general, but it seems to work for Rank 1 2 array broadcasted up to rank 3. For higher ranks a more rigorous proof is needed.
        let new_shape = self.dim.broadcast_shape(&shape)?;

        if new_shape.len() > R2::to_usize() {
            return Err(DimError::new("Array can not be broadcasted to shape"));
        } else {
            let n_old = helpers::multiply_list(&self.dim.shape, 1);
            let n = helpers::multiply_list(&new_shape.shape, 1);
            let repetitions = n / n_old;

            let mut new_data = vec![T::default(); n];
            for i in 0..repetitions {
                for j in 0..n_old {
                    new_data[i * n_old + j] = self.data[j].clone()
                }
            }

            return Ok(Ndarr {
                data: new_data,
                dim: new_shape,
            });
        }
    }

    pub fn broadcast<R2: Unsigned, D: Into<Dim<R2>>>(
        &self,
        shape: D,
    ) -> Result<Ndarr<T, Maximum<R, R2>>, DimError>
    where
        R: Max<R2>,
        <R as Max<R2>>::Output: Unsigned,
    {
        let shape = shape.into();
        let new_shape = self.dim.broadcast_shape(&shape)?;
        let n = helpers::multiply_list(&new_shape.shape, 1);

        let mut new_data = vec![T::default(); n];
        for i in 0..n {
            let indexes = new_shape.get_indexes(&i);
            let rev_casted_pos = Dim::<R>::rev_cast_pos(&self.dim, &indexes)?;
            new_data[i] = self.data[rev_casted_pos].clone();
        }
        Ok(Ndarr {
            data: new_data,
            dim: new_shape,
        })
    }

    pub fn broadcast_data<R2: Unsigned, D: Into<Dim<R2>>>(
        &self,
        shape: D,
    ) -> Result<Vec<T>, DimError>
    where
        R: Max<R2>,
        <R as Max<R2>>::Output: Unsigned,
    {
        let shape = shape.into();
        let new_shape = self.dim.broadcast_shape(&shape)?;

        let n = helpers::multiply_list(&new_shape.shape, 1);

        let mut new_data = vec![T::default(); n];
        for i in 0..n {
            let indexes = new_shape.get_indexes(&i);
            let rev_casted_pos = Dim::<R>::rev_cast_pos(&self.dim, &indexes)?;
            new_data[i] = self.data[rev_casted_pos].clone();
        }
        Ok(new_data)
    }

    fn t(&self) -> Self {
        let mut out_shape = self.dim.shape.clone();
        out_shape.reverse();
        let out_dim = Dim::<R>::new(&out_shape).unwrap();
        let mut out_arr = vec![T::default(); self.data.len()];
        for i in 0..self.data.len() {
            let new_indexes = self.dim.get_indexes(&i).reverse();
            let new_pos = out_dim.get_flat_pos(&new_indexes).unwrap();
            out_arr[new_pos] = self.data[i].clone();
        }
        Ndarr {
            data: out_arr,
            dim: out_dim,
        }
    }
}

pub fn de_slice<T: Clone + Debug + Default, R: Unsigned>(
    slices: &Vec<Ndarr<T, R>>,
    axis: usize,
) -> Ndarr<T, Add1<R>>
where
    R: Add<B1>,
    <R as Add<B1>>::Output: Unsigned,
{
    let l_slice = slices[0].len();
    let shape_slice = slices[0].dim.clone();

    let out_shape = shape_slice.clone().insert_element(axis, slices.len());
    let mut new_data: Vec<T> = vec![T::default(); helpers::multiply_list(&out_shape.shape, 1)];
    for i in 0..slices.len() {
        for j in 0..l_slice {
            //calculate the flat position of element j of slice i
            let ind = shape_slice.get_indexes(&j);
            //calculate the new flat position of element j of slice i
            let new_pos = out_shape
                .get_flat_pos(&ind.insert_element(axis, i))
                .unwrap();
            new_data[new_pos] = slices[i].data[j].clone()
        }
    }
    Ndarr {
        data: new_data,
        dim: out_shape,
    }
}

impl<T: Copy + Clone + Debug + Default> Ndarr<T, typenum::U0> {
    pub fn scalar(self) -> T {
        self.data[0]
    }
}

pub trait IntoNdarr<T, R: Unsigned>
where
    T: Debug + Clone + Default,
{
    fn into_ndarr(&self, shape: &Dim<R>) -> Ndarr<T, R>;
    fn get_rank(&self) -> usize;
}

impl<T, R: Unsigned> IntoNdarr<T, R> for Ndarr<T, R>
where
    T: Debug + Clone + Default,
{
    fn into_ndarr(&self, shape: &Dim<R>) -> Ndarr<T, R> {
        if self.dim.shape != *shape.shape {
            let err = format!(
                "self is shape {:?}, and ndarr is shape {:?}",
                self.dim.shape, shape.shape
            );
            panic!("Mismatch shape: {}", err)
        } else {
            self.clone()
        }
    }
    fn get_rank(&self) -> usize {
        R::to_usize()
    }
}

impl<T: Float + Clone + Debug + Default, R: Unsigned> Ndarr<T, R> {
    fn approx_epsilon<R2: Unsigned>(&self, other: Ndarr<T, R2>, epsilon: T) -> bool
    where
        R: Max<R2>,
        <R as typenum::Max<R2>>::Output: typenum::Unsigned,
        Ndarr<T, R>: Sub<Ndarr<T, R2>, Output = Ndarr<T, Maximum<R, R2>>>,
    {
        let diff = self.clone() - other;
        for val in diff.data {
            if val > epsilon {
                return false;
            }
        }
        true
    }

    fn approx<R2: Unsigned>(&self, other: &Ndarr<T, R2>) -> bool
    where
        R: Max<R2>,
        <R as typenum::Max<R2>>::Output: typenum::Unsigned,
        Ndarr<T, R>: Sub<Ndarr<T, R2>, Output = Ndarr<T, Maximum<R, R2>>>,
    {
        self.approx_epsilon(other.to_owned(), T::from(1e-10).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::*;
    use typenum::U2;

    #[test]
    fn constructor_test() {
        let arr = Ndarr::new(&[0, 1, 2, 3], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::from([[0, 1], [2, 3]]);
        assert_eq!(&arr.shape(), &[2, 2]);
        assert_eq!(&arr.rank(), &2);
        assert_eq!(&arr, &arr2)
    }
    #[test]
    fn bases() {
        let a: Ndarr<u32, U2> = Ndarr::zeros([2, 2]);
        let b: Ndarr<u32, U2> = Ndarr::ones([2, 2]);
        let c = Ndarr::fill(5, &[4]);
        assert_eq!(a, Ndarr::from([[0, 0], [0, 0]]));
        assert_eq!(b, Ndarr::from([[1, 1], [1, 1]]));
        assert_eq!(c, Ndarr::from([5, 5, 5, 5]));
    }
    #[test]
    fn bimap_test() {
        let arr1 = Ndarr::new(&[0, 1, 2, 3], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new(&[4, 5, 6, 7], [2, 2]).expect("Error initializing");
        assert_eq!(arr1.bimap(&arr2, |x, y| x + y).data, vec![4, 6, 8, 10])
    }

    #[test]
    fn transpose() {
        let arr = Ndarr::new(&[0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2]).expect("Error initializing");
        // same as arr.T.flatten() in numpy
        assert_eq!(arr.clone().t().data, vec![0, 4, 2, 6, 1, 5, 3, 7])
    }

    #[test]
    fn element_wise_ops() {
        let arr1 = Ndarr::new(&[1, 1, 1, 1], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new(&[1, 1, 1, 1], [2, 2]).expect("Error initializing");
        let arr3 = Ndarr::new(&[2, 2, 2, 2], [2, 2]).expect("Error initializing");
        assert_eq!((arr1.clone() + arr2.clone()).data, arr3.clone().data);
        assert_eq!((&arr1 - &arr2).data, vec![0, 0, 0, 0]);
        assert_eq!((&arr3 * &arr3).data, vec![4, 4, 4, 4]);
        assert_eq!((&arr3 / &arr3).data, vec![1, 1, 1, 1]);
        assert_eq!((-arr1).data, vec![-1, -1, -1, -1]);
    }
    #[test]
    fn assing_ops() {
        let mut arr = Ndarr::from([1, 2, 3]);
        arr += &1;
        arr += &Ndarr::from([-1, -1, -3]);
        assert_eq!(arr, Ndarr::from([1, 2, 1]))
    }

    #[test]
    fn broadcast_ops() {
        let a = Ndarr::from([[1, 2], [3, 4]]);
        let b = Ndarr::from([1, 2]);
        assert_eq!(&a + &b, Ndarr::from([[2, 4], [4, 6]]));
        assert_eq!(&b + &a, Ndarr::from([[2, 4], [4, 6]]))
    }

    #[test]
    fn scalar_ext() {
        let arr1 = Ndarr::new(&[2, 2, 2, 2], [2, 2]).expect("Error initializing");
        assert_eq!((&arr1 + 1).data, vec![3, 3, 3, 3]);
        assert_eq!((&arr1 - 2).data, vec![0, 0, 0, 0]);
        assert_eq!((&arr1 * 3).data, vec![6, 6, 6, 6]);
        assert_eq!((&arr1 / 2).data, vec![1, 1, 1, 1]);
    }

    #[test]
    fn slice_arr() {
        let arr = Ndarr::new(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [2, 3, 3],
        )
        .unwrap();
        // [[[ 0,  1,  2],
        //[ 3,  4,  5],
        //[ 6,  7,  8]],
        //-------------
        //[[ 9, 10, 11],
        //[12, 13, 14],
        //[15, 16, 17]]])
        let slices_0 = arr.clone().slice_at(0);
        let slices_1 = arr.clone().slice_at(1);
        let slices_2 = arr.clone().slice_at(2);

        assert_eq!(
            slices_0[0],
            Ndarr::new(&[0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3]).unwrap()
        );
        assert_eq!(
            slices_1[0],
            Ndarr::new(&[0, 1, 2, 9, 10, 11], [2, 3]).unwrap()
        );
        assert_eq!(
            slices_2[0],
            Ndarr::new(&[0, 3, 6, 9, 12, 15], [2, 3]).unwrap()
        );
    }

    #[test]
    fn deslice() {
        let arr = Ndarr::from([[1, 2], [3, 4]]);
        let slices0 = arr.slice_at(0);
        let slices1 = arr.slice_at(1);
        assert_eq!(arr, de_slice(&slices0, 0));
        assert_eq!(arr, de_slice(&slices1, 1));
    }
    #[test]
    fn scan() {
        let arr = Ndarr::from([[1, 2], [3, 4]]);
        let cumsum_r0 = arr.scanr(0, |x, y| x + y);
        let cumsum_r1 = arr.scanr(1, |x, y| x + y);
        let cumsum_l0 = arr.scanl(0, |x, y| x + y);
        let cumsum_l1 = arr.scanl(1, |x, y| x + y);
        assert_eq!(cumsum_r0, Ndarr::from([[1, 2], [4, 6]]));
        assert_eq!(cumsum_r1, Ndarr::from([[1, 3], [3, 7]]));
        assert_eq!(cumsum_l0, Ndarr::from([[4, 6], [3, 4]]));
        assert_eq!(cumsum_l1, Ndarr::from([[3, 2], [7, 4]]));
        //let arr2 = Ndarr::from([0.1, 0.2, 0.3]);
        //let sump = arr2.scanr(0, |x, y| x + y);
    }

    #[test]
    fn reduce() {
        let arr = Ndarr::new(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [2, 3, 3],
        )
        .unwrap();
        // [[[ 0,  1,  2],
        //[ 3,  4,  5],
        //[ 6,  7,  8]],
        //-------------
        //[[ 9, 10, 11],
        //[12, 13, 14],
        //[15, 16, 17]]])
        let red_0 = arr.clone().reduce(0, |x, y| x + y).unwrap();
        let red_1 = arr.reduce(1, |x, y| x + y).unwrap();
        assert_eq!(
            red_0,
            Ndarr::new(&[9, 11, 13, 15, 17, 19, 21, 23, 25], [3, 3]).unwrap()
        );
        assert_eq!(red_1, Ndarr::new(&[9, 12, 15, 36, 39, 42], [2, 3]).unwrap());
    }

    #[test]
    fn diatic_polymorphism() {
        let arr1 = Ndarr::from([[1, 2], [3, 4]]);
        let arr2 = Ndarr::from([1, 1]);
        assert_eq!(
            ops::poly_diatic(&arr2, &arr1, |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
        assert_eq!(
            ops::poly_diatic(&arr1, &arr2, |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
    }

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
        let r1 = outer_product(|x, y| x + y, &z, &z);
        let r2 = outer_product(g, &z, &z);

        assert_eq!(r1, Ndarr::from([[2, 3, 4], [3, 4, 5], [4, 5, 6]]));
        assert_eq!(r2, Ndarr::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
        //let c = Ndarr::from(["a", "b", "c", "d"]);
        //let d = Ndarr::from(["1", "2", "3", "4"]);
        //let ap = |x: &str, y: &str| (x.to_owned() + y);
        //let r3 = ops::outer_product(ap, &c, &d);
    }

    #[test]
    fn float_ops() {
        let a = Ndarr::from([0.1]);
        assert_eq!(a.sin(), Ndarr::from([0.1_f64.sin()]));
        assert_eq!(a.cos(), Ndarr::from([0.1_f64.cos()]));
        assert_eq!(a.tan(), Ndarr::from([0.1_f64.tan()]));
        assert_eq!(a.sinh(), Ndarr::from([0.1_f64.sinh()]));
        assert_eq!(a.cosh(), Ndarr::from([0.1_f64.cosh()]));
        assert_eq!(a.ln(), Ndarr::from([0.1_f64.ln()]));
        assert_eq!(a.log2(), Ndarr::from([0.1_f64.log2()]));
        assert_eq!(a.log(3.0), Ndarr::from([0.1_f64.log(3.0)]));
    }
    #[test]
    fn reshape() {
        //let a = Ndarr::from([1, 2, 3, 4]).reshape(&[2, 2]).unwrap();
        //assert_eq!(a, Ndarr::from([[1, 2], [3, 4]]))
    }

    #[test]
    fn ranges() {
        let a = Ndarr::from(0..4);
        assert_eq!(a, Ndarr::from([0, 1, 2, 3]))
    }

    #[test]
    fn abs() {
        let a = Ndarr::from([-1, -3, 4]);
        assert_eq!(a.abs(), Ndarr::from([1, 3, 4]))
    }
}
