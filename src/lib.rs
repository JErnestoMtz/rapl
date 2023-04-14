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
mod errors;
mod helpers;
mod natives;
mod scalars;
mod utils;
mod maps;
mod display;

#[cfg(feature = "complex")]
mod complex_tensor;
use std::{
    fmt::Debug,
    fmt::{Display},
};
#[cfg(feature = "complex")]
pub mod complex;


pub use scalars::{Scalar};
pub use errors::DimError;

pub use helpers::{broadcast_shape, const_max};

#[cfg(feature = "complex")]
pub use complex::*;

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
                shape,
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn from<P: Into<Self>>(p: P) -> Self {
        p.into()
    }

    pub fn reshape<const R2: usize>(&self, shape: &[usize; R2]) -> Result<Ndarr<T,R2>,DimError>
    where
        [usize; const_max(R, R2)]: Sized,
    {
        if helpers::multiply_list(&self.shape, 1) != helpers::multiply_list(shape, 1){
            return Err(DimError::new(&format!("Can not reshape array with shape {:?} to {:?}.",&self.shape, shape)))
        }
        Ok(Ndarr{data: self.data.clone(), shape: *shape})
    }

    pub fn slice_at(&self, axis: usize) -> Vec<Ndarr<T, { R - 1 }>>
    where
        [usize; R - 1]: Sized,
    {
        let n = helpers::multiply_list(&self.shape, 1); // number of elements in original array
        let new_shape: [usize; R - 1] = helpers::remove_element(self.shape, axis);
        let n_new_arrs = self.shape[axis]; // number of new arrays

        let iota = 0..n;

        let indexes: Vec<[usize; R]> = iota.map(|i| helpers::get_indexes(&i, &self.shape)).collect(); //indexes of each element

        let mut out: Vec<Ndarr<T, { R - 1 }>> = Vec::new(); // to sore

        for i in 0..n_new_arrs {
            let mut this_data: Vec<T> = Vec::new();
            for j in 0..n {
                if indexes[j][axis] == i {
                    let ind = helpers::get_flat_pos(&indexes[j], &self.shape).unwrap();
                    this_data.push(self.data[ind].clone())
                }
            }
            //TODO: remove push, with allocation size
            out.push(Ndarr::new(&this_data, new_shape).expect("Error initializing"))
        }
        out
    }

    pub fn reduce<F: Fn(T, T) -> T + Clone>(&self, axis: usize, f: F) -> Result<Ndarr<T, { R - 1 }>, DimError>
    where
        [usize; R - 1]: Sized,
    {
        if axis >= R {
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



    pub fn broadcast_to<const R2: usize>(&self, shape: &[usize; R2]) -> Result<Ndarr<T, { const_max(R, R2) }>, DimError>
    where
        [usize;  const_max(R, R2) ]: Sized,
    {
        //see https://numpy.org/doc/stable/user/basics.broadcasting.html
        //TODO: not sure at all if this implementation is general, but it seems to work for Rank 1 2 array broadcasted up to rank 3. For higher ranks a more rigorous proof is needed.
        let new_shape = helpers::broadcast_shape(&self.shape, shape)?;

        if new_shape.len() > R2 {
            return Err(DimError::new("Array can not be broadcasted to shape"));
        } else {
            let n_old = helpers::multiply_list(&self.shape, 1);
            let n = helpers::multiply_list(&new_shape, 1);
            let repetitions = n / n_old;

            let mut new_data = vec![T::default(); n];
            for i in 0..repetitions {
                for j in 0..n_old {
                    new_data[i * n_old + j] = self.data[j].clone()
                }
            }

            return Ok(Ndarr {
                data: new_data,
                shape: new_shape,
            });
        }
    }

    
    pub fn broadcast<const R2: usize>(&self, shape: &[usize; R2]) -> Result<Ndarr<T, { const_max(R, R2) }>, DimError>
    where
        [usize;  const_max(R, R2) ]: Sized,
    {
        let new_shape = helpers::broadcast_shape(&self.shape, shape)?;

        let n = helpers::multiply_list(&new_shape, 1);

        let mut new_data = vec![T::default(); n];
        for i in 0..n {
            let indexes = helpers::get_indexes(&i, &new_shape);
            let rev_casted_pos = helpers::rev_cast_pos(&self.shape, &indexes)?;
            new_data[i] = self.data[rev_casted_pos].clone();
        }
        Ok(Ndarr {
            data: new_data,
            shape: new_shape,
        })
    }

    pub fn broadcast_data<const R2: usize>(&self, shape: &[usize; R2]) -> Result<Vec<T>, DimError>
    where
        [usize; const_max(R, R2)]: Sized,
    {
        let new_shape = helpers::broadcast_shape(&self.shape, shape)?;

        let n = helpers::multiply_list(&new_shape, 1);

        let mut new_data = vec![T::default(); n];
        for i in 0..n {
            let indexes = helpers::get_indexes(&i, &new_shape);
            let rev_casted_pos = helpers::rev_cast_pos(&self.shape, &indexes)?;
            new_data[i] = self.data[rev_casted_pos].clone();
        }
        Ok(new_data)
    }

    fn t(self) -> Self {
        let shape = self.shape;
        let mut out_dim: [usize; R] = self.shape;
        out_dim.reverse();
        let mut out_arr = vec![T::default(); self.data.len()];
        for i in 0..self.data.len() {
            let mut new_indexes = helpers::get_indexes(&i, &shape);
            new_indexes.reverse();
            let new_pos = helpers::get_flat_pos(&new_indexes, &out_dim).unwrap();
            out_arr[new_pos] = self.data[i].clone();
        }
        Ndarr {
            data: out_arr,
            shape: out_dim,
        }
    }
}

pub fn de_slice<T: Clone + Debug + Default, const R: usize>(slices: &Vec<Ndarr<T,R>>, axis: usize)->Ndarr<T,{R+1}>{
    let l_slice = slices[0].len();
    let shape_slice = slices[0].shape();

    let out_shape : [usize; R + 1]= helpers::insert_element(shape_slice, axis, slices.len());
    let mut  new_data: Vec<T> = vec![T::default(); helpers::multiply_list(&out_shape, 1)];
    for i in 0..slices.len(){
        for j in 0..l_slice{
            //calculate the flat position of element j of slice i
            let ind = helpers::get_indexes(&j, &shape_slice);
            let new_pos = helpers::get_flat_pos(&helpers::insert_element( ind, axis, i), &out_shape).unwrap();
            new_data[new_pos] = slices[i].data[j].clone()
        }
    }
    Ndarr { data: new_data, shape: out_shape }
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
    fn into_ndarr(&self, shape: &[usize; R]) -> Ndarr<T, R>;
    fn get_rank(&self) -> usize;
}

impl<T, const R: usize> IntoNdarr<T, R> for Ndarr<T, R>
where
    T: Debug + Clone + Default,
{
    fn into_ndarr(&self, shape: &[usize; R]) -> Ndarr<T, R> {
        if self.shape != *shape {
            let err = format!(
                "self is shape {:?}, and ndarr is shape {:?}",
                self.shape, shape
            );
            panic!("Mismatch shape: {}", err)
        } else {
            self.clone()
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
    fn helpers(){
        assert_eq!(helpers::insert_element([1,2,3], 0, 0),[0,1,2,3])
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
    fn deslice(){
        let arr = Ndarr::from([[1,2],[3,4]]);
        let slices0 = arr.slice_at(0);
        let slices1 = arr.slice_at(1);
        assert_eq!(arr, de_slice(&slices0, 0));
        assert_eq!(arr, de_slice(&slices1, 1));
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
            ops::poly_diatic(&arr2, &arr1, |x, y| x + y).unwrap(),
            Ndarr::from([[2, 3], [4, 5]])
        );
        assert_eq!(
            ops::poly_diatic(&arr1, &arr2, |x, y| x + y).unwrap(),
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
    #[test]
    fn map_between_types_test() {
        let x: Ndarr<i16, 1> = Ndarr::from([1,2,3]);
        println!("x: {:#?}", x.data);
        println!("x shape: {:#?}", x.shape);
        let res: Ndarr<f32, 1> = x.map_types(|x| *x as f32);
        println!("res: {:#?}", res);
        assert!(res.data[0] == 1.0_f32);
        let a_string: Ndarr<String, 1> = x.map_types(|x| x.to_string());
        assert_eq!(a_string.data, vec!["1".to_owned(), "2".to_owned(), "3".to_owned()]);

        println!("a_string: {:#?}", a_string); // ["1","2","3"]
    }
}
