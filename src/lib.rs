//! **Enjoyable, composable, hackable N-dimensional arrays for Rust.**
//!
//! `rapl` combines the familiar parts of NumPy with array programming
//! languages like APL and BQN. Ranks are part of the type, so rank mistakes
//! are compile errors, and every operation is built from a small set of
//! composable primitives.
//!
//! ```
//! use rapl::*;
//!
//! let x = Ndarr::from([3, 1, 4, 1, 5, 9, 2, 6]);
//! let windows = x.slice(s![Win(3)])?;            // a [6, 3] view, no copy
//! let peaks = windows.reduce(1, i32::max)?;
//! assert_eq!(peaks, Ndarr::from([4, 4, 5, 9, 9, 9]));
//!
//! let a = Ndarr::from([1, 2, 3]);
//! let b = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
//! assert_eq!(a + b - 1, Ndarr::from([[1, 3, 5], [4, 6, 8]]));  // broadcasting
//!
//! println!("{peaks}");
//! // ┌→──────────┐
//! // │4 4 5 9 9 9│
//! // └~──────────┘
//! # Ok::<(), DimError>(())
//! ```
//!
//! `rapl` is in early development: the API is still settling and speed is not
//! yet a priority. See the [README](https://github.com/JErnestoMtz/rapl) for a
//! tour of the API.

mod array;
mod buffer;
mod core_ops;
mod display;
mod errors;
mod indexing;
mod natives;
pub mod ops;
mod scalars;
mod shape;

pub mod utils;

#[cfg(feature = "complex")]
pub mod complex;
#[cfg(feature = "complex")]
mod complex_tensor;

pub use array::{
    BumpRank, ElemIter, IndexInt, NdView, NdViewMut, Ndarr, NewAxis, Selector, Slice, SliceArgs,
    SliceSpec, Win,
};
pub use buffer::{Buffer, BufferMut};
pub use core_ops::ScanDirection;
pub use errors::DimError;
pub use scalars::Scalar;

#[cfg(feature = "complex")]
pub use complex::*;

pub use shape::{
    BroadcastRank, Broadcasted, CellAxes, ContractAxes, ContractRank, Contracted, Dim, Dyn,
    FrameRank, InsertAxisRank, Inserted, Keep, Rank, RankStore, ReduceAxis, RemoveAxisRank,
    Removed, StaticRank,
};

pub use typenum::{B0, B1, U0, U1, U2, U3, U4, U5, U6, U7, U8};

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::U2;

    fn stack<T: Clone, R: Rank + InsertAxisRank>(
        arrays: &[Ndarr<T, R>],
        axis: usize,
    ) -> Ndarr<T, Inserted<R>> {
        let expanded: Vec<_> = arrays
            .iter()
            .map(|array| array.insert_axis_view(axis).unwrap())
            .collect();
        Ndarr::concatenate(axis, &expanded).unwrap()
    }

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
        let c = Ndarr::fill(5, [4]);
        assert_eq!(a, Ndarr::from([[0, 0], [0, 0]]));
        assert_eq!(b, Ndarr::from([[1, 1], [1, 1]]));
        assert_eq!(c, Ndarr::from([5, 5, 5, 5]));
    }
    #[test]
    fn zip_with_same_shape() {
        let arr1 = Ndarr::new(&[0, 1, 2, 3], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new(&[4, 5, 6, 7], [2, 2]).expect("Error initializing");
        assert_eq!(
            arr1.zip_with(&arr2, |x, y| x + y).unwrap().data(),
            &[4, 6, 8, 10]
        )
    }

    #[test]
    fn transpose() {
        let arr = Ndarr::new(&[0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2]).expect("Error initializing");
        // same as arr.T.flatten() in numpy
        assert_eq!(arr.t().data(), &[0, 4, 2, 6, 1, 5, 3, 7])
    }

    #[test]
    fn element_wise_ops() {
        let arr1 = Ndarr::new(&[1, 1, 1, 1], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new(&[1, 1, 1, 1], [2, 2]).expect("Error initializing");
        let arr3 = Ndarr::new(&[2, 2, 2, 2], [2, 2]).expect("Error initializing");
        assert_eq!((arr1.clone() + arr2.clone()).data(), arr3.data());
        assert_eq!((&arr1 - &arr2).data(), &[0, 0, 0, 0]);
        assert_eq!((&arr3 * &arr3).data(), &[4, 4, 4, 4]);
        assert_eq!((&arr3 / &arr3).data(), &[1, 1, 1, 1]);
        assert_eq!((-arr1).data(), &[-1, -1, -1, -1]);
    }
    #[test]
    fn assing_ops() {
        let mut arr = Ndarr::from([1, 2, 3]);
        arr += 1;
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
        assert_eq!((&arr1 + 1).data(), &[3, 3, 3, 3]);
        assert_eq!((&arr1 - 2).data(), &[0, 0, 0, 0]);
        assert_eq!((&arr1 * 3).data(), &[6, 6, 6, 6]);
        assert_eq!((&arr1 / 2).data(), &[1, 1, 1, 1]);
    }

    #[test]
    fn slice_arr() {
        let arr = Ndarr::new(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [2, 3, 3],
        )
        .unwrap();
        let owned_slices = |axis: usize| -> Vec<Ndarr<i32, U2>> {
            (0..arr.shape()[axis])
                .map(|index| arr.index_axis_view(axis, index).unwrap().to_owned_array())
                .collect()
        };
        let slices_0 = owned_slices(0);
        let slices_1 = owned_slices(1);
        let slices_2 = owned_slices(2);

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
    fn stack_roundtrip() {
        let arr = Ndarr::from([[1, 2], [3, 4]]);
        let owned_slices = |axis: usize| -> Vec<Ndarr<i32, U1>> {
            (0..arr.shape()[axis])
                .map(|index| arr.index_axis_view(axis, index).unwrap().to_owned_array())
                .collect()
        };
        assert_eq!(arr, stack(&owned_slices(0), 0));
        assert_eq!(arr, stack(&owned_slices(1), 1));
    }

    #[test]
    fn scan() {
        let arr = Ndarr::from([[1, 2], [3, 4]]);
        let cumsum = |axis, direction| arr.scan_axis(axis, direction, |acc, x| acc + x).unwrap();
        assert_eq!(
            cumsum(0, ScanDirection::Forward),
            Ndarr::from([[1, 2], [4, 6]])
        );
        assert_eq!(
            cumsum(1, ScanDirection::Forward),
            Ndarr::from([[1, 3], [3, 7]])
        );
        assert_eq!(
            cumsum(0, ScanDirection::Backward),
            Ndarr::from([[4, 6], [3, 4]])
        );
        assert_eq!(
            cumsum(1, ScanDirection::Backward),
            Ndarr::from([[3, 2], [7, 4]])
        );
    }

    #[test]
    fn reduce() {
        let arr = Ndarr::new(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [2, 3, 3],
        )
        .unwrap();
        let red_0 = arr.reduce(0, |x, y| x + y).unwrap();
        let red_1 = arr.reduce(1, |x, y| x + y).unwrap();
        assert_eq!(
            red_0,
            Ndarr::new(&[9, 11, 13, 15, 17, 19, 21, 23, 25], [3, 3]).unwrap()
        );
        assert_eq!(red_1, Ndarr::new(&[9, 12, 15, 36, 39, 42], [2, 3]).unwrap());
    }

    #[test]
    fn dyadic_polymorphism() {
        let arr1 = Ndarr::from([[1, 2], [3, 4]]);
        let arr2 = Ndarr::from([1, 1]);
        assert_eq!(
            arr2.zip_with(&arr1, |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
        assert_eq!(
            arr1.zip_with(&arr2, |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
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
        let a = Ndarr::from([1, 2, 3, 4]).reshape([2, 2]).unwrap();
        assert_eq!(a, Ndarr::from([[1, 2], [3, 4]]))
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
    #[test]
    fn roll() {
        let a = Ndarr::from([[1, 2], [3, 4]]);
        assert_eq!(a.roll(1, 1), Ndarr::from([[2, 1], [4, 3]]))
    }
}
