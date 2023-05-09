use std::cmp::max;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

use crate::errors::DimError;
use crate::helpers::multiply_list;
use typenum::{Add1, Max, Maximum, Sub1, UTerm, Unsigned, B1, U1, U2, U3, U4, U5};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dim<R: Unsigned> {
    pub shape: Vec<usize>,
    rank: PhantomData<R>,
}

//<R as typenum::Unsigned>::to_usize(rank)
impl<R: Unsigned> Dim<R> {
    pub fn new(dim: &[usize]) -> Result<Self, DimError> {
        Ok(Dim {
            shape: dim.to_owned(),
            rank: PhantomData,
        })
    }
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    pub fn get_indexes(&self, n: &usize) -> Self {
        let r = self.shape.len();
        let mut ind = vec![0; r];
        let shape = self.shape.clone();
        for i in (0..r).rev() {
            let n_block = multiply_list(&shape[i + 1..], 1);
            ind[i] = ((n - (n % n_block)) / n_block) % shape[i]
        }
        Self::new(&ind).unwrap()
    }
    pub fn get_flat_pos(&self, indexes: &Self) -> Result<usize, DimError> {
        let mut ind = 0;
        let shape = self.shape.clone();
        let r = self.shape.len();
        let indexes = indexes.shape.clone();
        for i in 0..r {
            if indexes[i] >= shape[i] {
                return Err(DimError::new("Index out of bounds"));
            }
            ind += indexes[r - i - 1] * multiply_list(&shape[r - i..], 1);
        }
        Ok(ind)
    }
    pub fn remove_element(self, index: usize) -> Dim<Sub1<R>>
    where
        R: Sub<B1>,
        <R as Sub<B1>>::Output: Unsigned,
    {
        let r = R::to_usize();
        let mut data = self.shape.clone();
        assert!(index < r);
        data.remove(index);
        Dim {
            shape: data,
            rank: PhantomData,
        }
    }
    pub fn remove_element_notyped(self, index: usize) -> Dim<UTerm> {
        let r = self.shape.len();
        let mut data = self.shape.clone();
        assert!(index < r);
        data.remove(index);
        Dim {
            shape: data,
            rank: PhantomData,
        }
    }
    pub fn insert_element(self, index: usize, element: usize) -> Dim<Add1<R>>
    where
        R: Add<B1>,
        <R as Add<B1>>::Output: Unsigned,
    {
        let mut result = self.shape.clone();
        result.insert(index, element);
        Dim::<Add1<R>>::new(&result).unwrap()
    }

    ///Paths a shape of rank R with ones in the left until is rank R2.
    pub fn path_shape<R2: Unsigned>(&self) -> Result<Dim<R2>, DimError> {
        let r1 = R::to_usize();
        let r2 = R2::to_usize();
        if R::to_usize() > R2::to_usize() {
            return Err(DimError::new(&format!(
                "Can not path shape {:?} of rank {} to rank {}.",
                self.shape, r1, r2
            )));
        } else {
            let mut out = vec![1; r2];
            for i in 0..r1 {
                out[r2 - i - 1] = self.shape[r1 - i - 1]
            }
            Ok(Dim::<R2>::new(&out)?)
        }
    }

    pub fn rev_cast_pos<R1: Unsigned, R2: Unsigned>(
        small_shape: &Dim<R1>,
        indexes: &Dim<R2>,
    ) -> Result<usize, DimError> {
        let r1 = R1::to_usize();
        let r2 = R2::to_usize();
        let mut indexes = indexes.shape.clone();
        //paths shape with 1 on the left until is the same shape as indexes;
        let padded = small_shape.path_shape::<R2>()?.shape;
        //initialize the indexes, with the same rank of small_shape
        let mut rev_cast_ind = vec![0; r1];
        for i in 0..r2 {
            //Check if index is greater or equal than shape
            if padded[i] <= indexes[i] {
                //if it is set to max index i.e padded[i] -1
                indexes[i] = padded[i] - 1
            }
        }
        for i in 0..r1 {
            rev_cast_ind[r1 - i - 1] = indexes[r2 - i - 1]
        }
        let rev_cast_ind = Dim::<R1>::new(&rev_cast_ind)?;
        Ok(small_shape.get_flat_pos(&rev_cast_ind)?)
    }
    fn index_or(arr: &[usize], index: usize, or: usize) -> usize {
        if index >= arr.len() {
            or
        } else {
            arr[index]
        }
    }
    pub fn broadcast_shape<R2: Unsigned>(
        &self,
        other: &Dim<R2>,
    ) -> Result<Dim<Maximum<R, R2>>, DimError>
    where
        R: Max<R2>,
        <R as Max<R2>>::Output: Unsigned,
    {
        let r1 = R::to_usize();
        let r2 = R2::to_usize();
        let mut out_shape = vec![0; max(r1, r2)];
        //get both shapes
        let mut sh1 = self.shape.clone();
        let mut sh2 = other.shape.clone();
        sh1.reverse();
        sh2.reverse();

        let l = max(r1, r2);
        for i in 0..l {
            let size1 = Self::index_or(&sh1, i, 1);
            let size2 = Self::index_or(&sh2, i, 1);
            //broadcasting criteria
            if size1 != 1 && size2 != 1 && size1 != size2 {
                return Err(DimError::new(&format!(
                    "Error arrays with shape {:?} and {:?} can not be broadcasted.",
                    self.shape, other.shape
                )));
            }
            out_shape[l - i - 1] = max(size1, size2)
        }
        Ok(Dim::<Maximum<R, R2>>::new(&out_shape)?)
    }
    pub fn len(&self) -> usize {
        self.shape.len()
    }
    pub fn reverse(&self) -> Self {
        let mut shape = self.shape.clone();
        shape.reverse();
        Dim {
            shape: shape,
            rank: PhantomData,
        }
    }
}

impl<R: Unsigned> From<&Dim<R>> for Dim<R> {
    fn from(value: &Dim<R>) -> Self {
        Dim {
            shape: value.shape.clone(),
            rank: PhantomData,
        }
    }
}
macro_rules! arr_to_dim {
    ($n:expr, $t:ty) => {
        impl From<[usize; $n]> for Dim<$t> {
            fn from(value: [usize; $n]) -> Self {
                Dim {
                    shape: value.to_vec(),
                    rank: PhantomData,
                }
            }
        }

        impl From<&[usize; $n]> for Dim<$t> {
            fn from(value: &[usize; $n]) -> Self {
                Dim {
                    shape: value.to_vec(),
                    rank: PhantomData,
                }
            }
        }
    };
}

arr_to_dim!(1, U1);
arr_to_dim!(2, U2);
arr_to_dim!(3, U3);
arr_to_dim!(4, U4);
arr_to_dim!(5, U5);

impl<const N1: usize, const N2: usize, const N3: usize> From<[[[usize; N1]; N2]; N3]> for Dim<U3> {
    fn from(value: [[[usize; N1]; N2]; N3]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2 * N3);
        for row in value.iter() {
            for column in row.iter() {
                data.extend_from_slice(column)
            }
        }

        Dim {
            shape: data,
            rank: PhantomData,
        }
    }
}

impl<const N1: usize, const N2: usize, const N3: usize> From<&[[[usize; N1]; N2]; N3]> for Dim<U3> {
    fn from(value: &[[[usize; N1]; N2]; N3]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2 * N3);
        for row in value.iter() {
            for column in row.iter() {
                data.extend_from_slice(column)
            }
        }

        Dim {
            shape: data,
            rank: PhantomData,
        }
    }
}

#[cfg(test)]
mod dim_tests {
    use super::*;
    use typenum::{U2, U3, U5};
    #[test]
    pub fn init_dim() {
        assert!(Dim::<U3>::new(&[1, 2, 4]).is_ok());
        assert!(Dim::<U3>::new(&[1, 2]).is_err());
        assert!(Dim::<U5>::new(&[1, 2, 3, 4, 5]).is_ok());
    }
    #[test]
    pub fn get_ind() {
        let s = Dim::<U2>::new(&[2, 2]).unwrap();
        assert_eq!(s.get_indexes(&0).shape, vec![0, 0]);
        assert_eq!(s.get_indexes(&3).shape, vec![1, 1]);
        assert_eq!(s.get_indexes(&2).shape, vec![1, 0]);
    }

    #[test]
    pub fn rem_elm() {
        let s1 = Dim::<U3>::new(&[1, 2, 3]).unwrap();
        let s2 = Dim::<U2>::new(&[1, 3]).unwrap();
        assert_eq!(s1.remove_element(1), s2);
    }

    #[test]
    pub fn ins_elm() {
        let s1 = Dim::<U2>::new(&[1, 3]).unwrap();
        let s2 = Dim::<U3>::new(&[1, 2, 3]).unwrap();
        assert_eq!(s1.insert_element(1, 2), s2);
    }

    #[test]
    pub fn tpath_shape() {
        let s1 = Dim::<U2>::new(&[2, 3]).unwrap();
        let s2 = Dim::<U5>::new(&[1, 1, 1, 2, 3]).unwrap();
        assert_eq!(s1.path_shape().unwrap(), s2);
    }
}
