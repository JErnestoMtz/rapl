#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(const_trait_impl)]
#![feature(const_cmp)]


mod helpers;
mod ops;
mod primitives;
mod scalars;
mod natives;
use core::slice;
use std::{
    fmt::Debug,
    fmt::{write, Display},
    ops::Deref,
};

use scalars::Scalar;

// main struct of N Dimensional generic array.
//the shape is denoted by the `shape` array where the length is the Rank of the Ndarray
//the actual values are stored in a flattened state in a rank 1 array

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ndarr<T: Copy + Clone + Default, const R: usize> {
    pub data: Vec<T>,
    pub shape: [usize; R],
}

#[derive(Debug, Copy, Clone)]
pub struct Ndarr2<T: Copy + Clone + Default, const N: usize, const SHAPE: &'static [usize]> {
    pub data: [T; N],
}

impl<T: Copy + Clone + Debug + Default, const R: usize> Ndarr<T, R> {
    //TODO: implement errors
    pub fn new(data: &[T], shape: [usize; R]) -> Result<Self, String> {
        let n = helpers::multiply_list(&shape, 1);
        if data.len() == n {
            Ok(Ndarr {
                data: data.to_vec(),
                shape: shape,
            })
        } else {
            Err(format!(
                "The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.",
                shape,
                n,
                data.len()
            ))
        }
    }
    pub fn rank(&self) -> usize {
        R
    }
    pub fn shape(&self) -> [usize; R] {
        self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn from<P: Into<Self>>(p: P)->Self{
        p.into()
    }
}


//impl <T: Copy + Debug + Clone + Default, const R: usize> Rank for Ndarr<T,R> {
    //fn get_rank(&self)->usize {
        //R
    //}
    
//}
//impl<T: Copy + Clone + Debug + Default, const N: usize, const SHAPE: &'static [usize]> Ndarr2<T, N, SHAPE> {
////TODO: implement errors
//pub fn new(data: [T; N]) -> Result<Self, String> {
//let n = helpers::multiply_list(SHAPE, 1);
//if data.len() == n {
//Ok(Ndarr2 {
//data: data,
//})
//} else {
//Err(format!(
//"The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.",
//SHAPE, n, N
//))
//}
//}
//pub fn rank(self) -> usize {
//SHAPE.len()
//}
//pub fn shape(self) -> &'static [usize] {
//SHAPE
//}
//}

impl<T: Copy + Clone + Debug + Default + Display, const R: usize> Display for Ndarr<T, R> {
    // Kind of nasty function, it can be imprube a lot, but I think there is no scape from recursion.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //convert to string
        let strs: Vec<String> = self.data.iter().map(|x| x.to_string()).collect();
        // len of each strings
        let binding: Vec<usize> = strs.clone().iter().map(|s| s.len()).collect();
        // max len ( for formatting)
        let max_size = binding.iter().max().unwrap();
        //format each string
        let mut fmt_str: Vec<String> = strs
            .iter()
            .map(|s| helpers::format_vla(s.to_string(), max_size))
            .collect();

        let mut splits = self.shape.clone();
        //splits.reverse();

        fn slip_format<'a>(strings: &'a mut [String], splits: &'a [usize]) -> () {
            if splits.len() == 0 {
                return;
            }
            let l = helpers::multiply_list(splits, 1);
            let n_splits = strings.len() / l;
            for i in 0..n_splits {
                let new_s: &mut [String] = &mut strings[i * l..(i + 1) * l];
                new_s[0].insert_str(0, "[");
                new_s[l - 1].push_str("]");
                slip_format(new_s, &splits[1..]);
            }
            return;
        }
        // TODO: add new lines in the correct places to display it more numpy like
        slip_format(&mut fmt_str[0..], &mut splits[..]);

        let out = fmt_str.clone().join(" ");
        write!(f, "Ndarr({})", out)
    }
}

//impl<T: Copy + Clone + Debug + Default + Display, const N: usize, const SHAPE: &'static [usize]> Display for Ndarr2<T, N, SHAPE>

//{
//// Kind of nasty function, it can be imprube a lot, but I think there is no scape from recursion.
//fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
//{
//let strs = self.data.map(|x| x.to_string());
//let binding = strs.clone().map(|s| s.len());
//let max_size = binding.iter().max().unwrap();
//let mut fmt_str: [String; N] = strs.map(|s| helpers::format_vla(s, max_size));
//let r: usize = SHAPE.len();
//let mut splits = Vec::with_capacity(r);

////reverse array to simplify code
//splits.extend(SHAPE.iter().rev());

//fn slip_format<'a>(strings: &'a mut [String], splits: &'a [usize]) -> () {
//if splits.len() == 0 {
//return;
//}
//let l = helpers::multiply_list(splits, 1);
//let n_splits = strings.len() / l;
//for i in 0..n_splits {
//let new_s: &mut [String] = &mut strings[i * l..(i + 1) * l];
//new_s[0].insert_str(0, "[");
//new_s[l - 1].push_str("]");
//slip_format(new_s, &splits[1..]);
//}
//return;
//}
//// TODO: add new lines in the correct places to display it more numpy like
//slip_format(&mut fmt_str[0..], &mut splits[..]);

//let out = fmt_str.clone().join(" ");
//write!(f, "Ndarr({})", out)
//}
//}
trait Bimap<F> {
    fn bimap(self, other: Self, f: F) -> Self;
    fn bimap_in_place(&mut self, other: Self, f: F);
}

///////////

//TODO: the problem her is we can not use Into because we need to know the shape, and Into trait does not passes any reference
pub trait IntoNdarr<T, const R: usize>
where
    T: Debug + Copy + Clone + Default,
{
    fn into_ndarr(self, shape: &[usize; R]) -> Ndarr<T, R>;
    fn get_rank(&self)->usize;
}

//pub trait IntoNdarr2<T, const N: usize, const SHAPE: &'static [usize]>
//where T: Debug + Copy + Clone + Default,
//[T; N]: Default
//{
//fn into_ndarr2(self) -> Ndarr2<T,N,SHAPE>;
//}

/////////

impl<T, const R: usize> IntoNdarr<T, R> for Ndarr<T, R>
where
    T: Debug + Copy + Clone + Default,
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
    fn get_rank(&self)->usize {
        R
    }
}



//impl<T, const N: usize, const  SHAPE: &'static [usize]> IntoNdarr2<T,N,SHAPE> for Ndarr2<T,N,SHAPE>
//where T: Debug + Copy + Clone + Default,
//[T; N]: Default
//{
//fn into_ndarr2(self) -> Ndarr2<T,N,SHAPE> {
//self
//}
//}

////////

// Here we need to think about if valueble maybe checking for the same shape and return an option instead
impl<F, T: Copy + Debug + Clone + Default, const R: usize> Bimap<F> for Ndarr<T, R>
where
    F: Fn(&T, &T) -> T,
{
    fn bimap(self, other: Self, f: F) -> Self {
        let mut out = vec![T::default(); self.data.len()];
        for i in 0..out.len() {
            out[i] = f(&self.data[i], &other.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }

    fn bimap_in_place(&mut self, other: Self, f: F) {
        for i in 0..self.data.len() {
            self.data[i] = f(&self.data[i], &other.data[i])
        }
    }
}

//impl<F, T: Copy + Debug + Clone + Default, const N: usize, const SHAPE: &'static [usize]> Bimap<F>
//for Ndarr2<T, N, SHAPE>
//where
//F: Fn(&T, &T) -> T,
//[T; N]: Default,
//{
//fn bimap(self, other: Self, f: F) -> Self {
//let mut out: [T; N] = Default::default();
//for i in 0..N {
//out[i] = f(&self.data[i], &other.data[i])
//}
//Ndarr2 {
//data: out,
//}
//}
//}

/////

trait Map<F> {
    fn map(self, f: F) -> Self;

    fn map_in_place(&mut self, f: F);
}

// Here we need to think about if worth it maybe checking for the same shape and return an option instead or just panic()
impl<F, T: Copy + Debug + Clone + Default, const R: usize> Map<F> for Ndarr<T, R>
where
    F: Fn(&T) -> T,
{
    fn map(self, f: F) -> Self {
        let mut out = vec![T::default(); self.data.len()];
        for i in 0..out.len() {
            out[i] = f(&self.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
    fn map_in_place(&mut self, f: F) {
        for i in 0..self.data.len() {
            self.data[i] = f(&self.data[i])
        }
    }
}



trait Transpose {
    fn t(self) -> Self;
}

// Generic transpose for array of rank R
// the basic idea of a generic transpose of an N-dimensional array is to flip de shape of it like in a mirror.
// The helper functions use in here can be derive with some maths, but maybe there is a better way to do it.
impl<T: Default + Copy + Clone, const R: usize> Transpose for Ndarr<T, R>
{
    fn t(self) -> Self {
        let shape = self.shape.clone();
        let mut out_dim: [usize; R] = self.shape.clone();
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

///////////////////////////////////////////
///
///
///
/// 



#[cfg(test)]
mod tests {
    use crate::primitives::{Slice, Reduce, Broadcast};

    use super::*;

    #[test]
    fn constructor_test() {
        let arr = Ndarr::new(&[0, 1, 2, 3], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::from([[0,1],[2,3]]);
        assert_eq!(&arr.shape(), &[2, 2]);
        assert_eq!(&arr.rank(), &2);
        assert_eq!(&arr, &arr2)
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
    fn scalar_ext() {
        let arr1 = Ndarr::new(&[2, 2, 2, 2], [2, 2]).expect("Error initializing");
        assert_eq!((&arr1 + &1).data, vec![3, 3, 3, 3]);
        assert_eq!((&arr1 - &2).data, vec![0, 0, 0, 0]);
        assert_eq!((&arr1 * &3).data, vec![6, 6, 6, 6]);
        assert_eq!((&arr1 / &2).data, vec![1, 1, 1, 1]);
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

        assert_eq!(slices_0[0], Ndarr::new(&[0,1,2,3,4,5,6,7,8], [3,3]).unwrap());
        assert_eq!(slices_1[0], Ndarr::new(&[0,1,2,9,10,11], [2,3]).unwrap());
        assert_eq!(slices_2[0], Ndarr::new(&[0,3,6,9,12,15], [2,3]).unwrap());
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
        assert_eq!(red_0, Ndarr::new(&[9,11,13,15,17,19,21,23,25], [3,3]).unwrap());
        assert_eq!(red_1, Ndarr::new(&[9,12,15,36,39,42], [2,3]).unwrap());
    }

    #[test]
    fn broadcast(){
        // see https://numpy.org/doc/stable/user/basics.broadcasting.html
        assert_eq!(helpers::broadcast_shape(&[2,2], &[2]).unwrap(),[2,2]);
        assert_eq!(helpers::broadcast_shape(&[2], &[2,2]).unwrap(),[2,2]);
        assert_eq!(helpers::broadcast_shape(&[3,3], &[3,3]).unwrap(),[3,3]);
        assert!(helpers::broadcast_shape(&[2,2], &[3,2,2]).is_ok());

        assert!(helpers::broadcast_shape(&[2,2], &[2,3]).is_err());
        assert!(helpers::broadcast_shape(&[2,2], &[2,2,3]).is_err());

        assert!(Ndarr::from([[1,2],[3,4]]).broadcast_to(&[2]).is_err());
        assert!(Ndarr::from([[1,2],[3,4]]).broadcast_to(&[4,2,2]).is_ok());
        assert!(Ndarr::from([[1,2],[3,4]]).broadcast_to(&[2,2]).is_ok());

        let a = Ndarr::new(&[1,2], [2]).unwrap();
        assert_eq!(a.broadcast(&[2,2]).unwrap(), Ndarr::new(&[1,2,1,2], [2,2]).unwrap());

    }

    #[test]
    fn diatic_polymorphism(){
        let arr1 = Ndarr::from([[1,2],[3,4]]);
        let arr2 = Ndarr::from([1,1]);
        assert_eq!(ops::poly_diatic(arr2.clone(), arr1.clone(), |x,y| x+y).unwrap(),Ndarr::from([[2,3],[4,5]]));
        assert_eq!(ops::poly_diatic(arr1, arr2, |x,y| x+y).unwrap(),Ndarr::from([[2,3],[4,5]]));

    }

    #[test]
    fn inner(){
        let x = Ndarr::from([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]);
        let y = Ndarr::from([[[1]],[[2]],[[3]]]);

        //let c = a.clone().t().broadcast(&[1,3,2]).unwrap().t();
        //let d = b.clone().broadcast(&[1,3,2]).unwrap();
        //let cc = c.broadcast(&[2,3,2]).unwrap();
        //let r = ops::poly_diatic(cc,d, |x,y| x*y).unwrap();
        //let rr = r.reduce(1, |x,y| x+y).unwrap();
        let r = ops::mat_mul(x, y);
        //print!("{:?}",&r.shape);
        //let a = Ndarr::from([1,2,3,4,5]);
        //let b = a.reduce(0, |x,y| x+y).unwrap();
        //let b = ops::mat_mul(a.clone(), a.t());
        println!("{:?}", r);


    }
}

