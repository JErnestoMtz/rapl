use std::marker::PhantomData;

use typenum::Unsigned;
use crate::{shape::Dim, helpers};

use super::*;


pub struct ArrayView<'a, T, R1: Unsigned> {
    data: *const T,
    shape: Dim<R1>,
    strides: Vec<usize>, 
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, R1: Unsigned> ArrayView<'a, T,R1> {
    fn new(data: *const T, shape: Dim<R1>) -> ArrayView<'a, T,R1> {
        let strides = Self::calculate_strides(&shape.shape);
        ArrayView {
            data,
            shape,
            strides,
            _phantom: PhantomData,
        }
    }
    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        let mut strides = vec![0; ndim];

        strides[ndim - 1] = 1;
        for i in (0..(ndim - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    strides
    }
    fn get_index<D: Into<Dim<R1>>>(&self, indices: D) -> usize {
        let indices: Dim<R1> = indices.into();
        let mut offset = 0;
        for i in 0..self.shape.len() {
            debug_assert!(indices.shape[i] < self.shape.shape[i]);
            offset += indices.shape[i] * self.strides[i];
        }
        offset
    }
    fn get_raw<D: Into<Dim<R1>>>(&self, indices: D) -> *const T {
        let shape: Dim<R1> = indices.into();
        let offset = self.get_index(&shape);
        unsafe { self.data.offset(offset as isize) }
    }


}

impl<T: Clone, R: Unsigned> Ndarr<T,R> {
    pub fn view<'a>(&'a self)->ArrayView<'a,T,R>{
        assert!(self.data.len() >= helpers::multiply_list(&self.dim.shape, 1));

        ArrayView::new(self.data.as_ptr(), self.dim.clone())
    }
    
}

#[cfg(test)]

mod views{
    use super::*;
    #[test]
    fn complete_view(){
        let a = Ndarr::from([[1,2,3],[4,5,6]]);
        let view = a.view();
        let v = view.get_raw(&[1,1]);
        let element = unsafe { *v };
    }
}