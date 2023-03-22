#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]

//!*NOTE*: `rapl`  requires Nightly and is strictly intended for non-production purposes only. `rapl` utilizes certain unstable features that may result in unexpected behavior, and is not optimized for performance.
//!`rapl` is an experimental numerical computing Rust that provides an simple way of working with N-dimensional array, along with a wide range of mathematical functions to manipulate them. It takes inspiration from NumPy and APL, with the primary aim of achieving maximum ergonomic and user-friendliness while maintaining generality. Notably, it offers automatic Rank Polymorphic broadcasting between arrays of varying shapes and scalars as a built-in feature.

//!```
//!#![feature(generic_const_exprs)]
//!use rapl::*;
//!fn main() {
//!     let a = Ndarr::from([1, 2, 3]);
//!     let b = Ndarr::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
//!     let r = a + b - 1;
//!     assert_eq!(r, Ndarr::from([[1, 3, 5], [4, 6, 8], [7, 9, 11]]));
//!}
//!```


pub mod ops;
mod helpers;
mod natives;
mod primitives;
mod scalars;
mod utils;
mod maps;
mod display;
use std::{
    fmt::Debug,
    fmt::{Display},
};


pub use primitives::DimError;
pub use scalars::{Scalar, Trig};
pub use helpers::{broadcast_shape, const_max};
pub use primitives::{Broadcast, Reduce, Slice, Reshape, Transpose};
pub use maps::{Bimap, Map};


///Main struct of N Dimensional generic array. The shape is denoted by the `shape` array where the length is the Rank of the Ndarray the actual values are stored in a flattened state in a rank 1 array.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ndarr<T: Clone + Default, const R: usize> {
    pub data: Vec<T>,
    pub shape: [usize; R],
}



impl<T: Clone + Debug + Default, const R: usize> Ndarr<T, R> {
    pub fn new(data: &[T], shape: [usize; R]) -> Result<Self, DimError> {
        let n = helpers::multiply_list(&shape, 1);
        if data.len() == n {
            Ok(Ndarr {
                data: data.to_vec(),
                shape: shape,
            })
        } else {
            Err(DimError::new(&format!(
                "The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.",
                shape,
                n,
                data.len()
            )))
        }
    }
    pub fn rank(&self) -> usize {
        R
    }
    pub fn shape(&self) -> [usize; R] {
        self.shape
    }
    pub fn flatten(self)->Vec<T>{
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn from<P: Into<Self>>(p: P) -> Self {
        p.into()
    }
}

impl<T: Copy + Clone + Debug + Default, const R: usize> Ndarr<T, R> {
    pub fn scalar(self) -> T {
        if R == 0 {
            self.data[0]
        } else {
            panic!("Can not convert {:?} to Scalar", self)
        }
    }
}



pub trait IntoNdarr<T, const R: usize>
where
    T: Debug + Clone + Default,
{
    fn into_ndarr(self, shape: &[usize; R]) -> Ndarr<T, R>;
    fn get_rank(&self) -> usize;
}

impl<T, const R: usize> IntoNdarr<T, R> for Ndarr<T, R>
where
    T: Debug + Clone + Default,
{
    fn into_ndarr(self, shape: &[usize; R]) -> Ndarr<T, R> {
        if self.shape != *shape {
            let err = format!(
                "self is shape {:?}, and ndarr is shape {:?}",
                self.shape, shape
            );
            panic!("Mismatch shape: {}", err)
        } else {
            self
        }
    }
    fn get_rank(&self) -> usize {
        R
    }
}





#[cfg(test)]
mod tests {

    use super::*;
    use ops::*;

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
        let a: Ndarr<u32, 2> = Ndarr::zeros(&[2, 2]);
        let b: Ndarr<u32, 2> = Ndarr::ones(&[2, 2]);
        let c = Ndarr::fill(5, &[4]);
        assert_eq!(a, Ndarr::from([[0, 0], [0, 0]]));
        assert_eq!(b, Ndarr::from([[1, 1], [1, 1]]));
        assert_eq!(c, Ndarr::from([5, 5, 5, 5]));
    }

    #[test]
    fn bimap_test() {
        let arr1 = Ndarr::new(&[0, 1, 2, 3], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new(&[4, 5, 6, 7], [2, 2]).expect("Error initializing");
        assert_eq!(arr1.bimap(arr2, |x, y| x + y).data, vec![4, 6, 8, 10])
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
        let red_1: Ndarr<i32, 2> = arr.reduce(1, |x, y| x + y).unwrap();
        assert_eq!(
            red_0,
            Ndarr::new(&[9, 11, 13, 15, 17, 19, 21, 23, 25], [3, 3]).unwrap()
        );
        assert_eq!(red_1, Ndarr::new(&[9, 12, 15, 36, 39, 42], [2, 3]).unwrap());
    }

    #[test]
    fn broadcast() {
        // see https://numpy.org/doc/stable/user/basics.broadcasting.html
        assert_eq!(helpers::broadcast_shape(&[2, 2], &[2]).unwrap(), [2, 2]);
        assert_eq!(helpers::broadcast_shape(&[2], &[2, 2]).unwrap(), [2, 2]);
        assert_eq!(helpers::broadcast_shape(&[3, 3], &[3, 3]).unwrap(), [3, 3]);
        assert!(helpers::broadcast_shape(&[2, 2], &[3, 2, 2]).is_ok());

        assert!(helpers::broadcast_shape(&[2, 2], &[2, 3]).is_err());
        assert!(helpers::broadcast_shape(&[2, 2], &[2, 2, 3]).is_err());

        assert!(Ndarr::from([[1, 2], [3, 4]]).broadcast_to(&[2]).is_err());
        assert!(Ndarr::from([[1, 2], [3, 4]])
            .broadcast_to(&[4, 2, 2])
            .is_ok());
        assert!(Ndarr::from([[1, 2], [3, 4]]).broadcast_to(&[2, 2]).is_ok());

        let a = Ndarr::new(&[1, 2], [2]).unwrap();
        assert_eq!(
            a.broadcast(&[2, 2]).unwrap(),
            Ndarr::new(&[1, 2, 1, 2], [2, 2]).unwrap()
        );
    }

    #[test]
    fn diatic_polymorphism() {
        let arr1 = Ndarr::from([[1, 2], [3, 4]]);
        let arr2 = Ndarr::from([1, 1]);
        assert_eq!(
            ops::poly_diatic(arr2.clone(), arr1.clone(), |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
        assert_eq!(
            ops::poly_diatic(arr1, arr2, |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
    }

    #[test]
    fn inner() {
        let x = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
        let y = Ndarr::from([[1, 2], [3, 4], [5, 6]]);
        let z = Ndarr::from([1, 2, 3, 4, 5]);
        let matmul = inner_closure(|x, y| x * y, |x, y| x + y);
        let r = matmul(x, y);
        let matmul = inner_closure(|x, y| x * y, |x, y| x + y);
        let r2 = matmul(z.clone(), z);
        let g1 = Ndarr::from("gattaca");
        let g2 = Ndarr::from("tattcag");
        let g = |a: char, b: char| {
            if a == b {
                1
            } else {
                0
            }
        };
        let numequals = inner_closure(g, |x, y| x + y);
        let ttt = numequals(g1, g2).scalar();

        assert_eq!(r, Ndarr::from([[22, 28], [49, 64]]));
        assert_eq!(r2.scalar(), 55);
        assert_eq!(ttt, 3)

        //println!("{:?}", r);
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
        let r1 = outer_product(|x, y| x + y, z.clone(), z.clone());
        let r2 = outer_product(g, z.clone(), z);

        assert_eq!(r1, Ndarr::from([[2, 3, 4], [3, 4, 5], [4, 5, 6]]));
        assert_eq!(r2, Ndarr::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
        //let c = Ndarr::from(["a","b","c","d"]);
        //let d = Ndarr::from(["1","2","3","4"]);
        //let ap = |x: &str, y: &str| (x.to_owned() + y);
        //let r3 = ops::outer( ap, c, d);
    }

    #[test]
    fn trig() {
        let a = Ndarr::from([0.1, 0.2]);
        assert_eq!(a.sin(), Ndarr::from([0.1_f64.sin(), 0.2_f64.sin()]))
    }
    #[test]
    fn reshape(){
        let a = Ndarr::from([1,2,3,4]).reshape(&[2,2]).unwrap();
        assert_eq!(a, Ndarr::from([[1,2],[3,4]]))

    }

    #[test]
    fn ranges(){
        let a = Ndarr::from(0..4);
        assert_eq!(a, Ndarr::from([0,1,2,3]))
    }

    #[test]
    fn abs(){
        let a = Ndarr::from([-1, -3, 4]);
        assert_eq!(a.abs(), Ndarr::from([1,3,4]))
    }
}
