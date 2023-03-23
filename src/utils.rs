use crate::helpers::multiply_list;

use super::*;

//// zeros and ones
impl<T, const R: usize> Ndarr<T, R>
where
    T: Clone + Debug + Default + From<u8>,
{
    pub fn zeros(shape: &[usize; R]) -> Self {
        let data: Vec<T> = vec![T::from(0); multiply_list(shape, 1)];
        Ndarr {
            data,
            shape: shape.clone(),
        }
    }

    pub fn ones(shape: &[usize; R]) -> Self {
        let data: Vec<T> = vec![T::from(1); multiply_list(shape, 1)];
        Ndarr {
            data,
            shape: shape.clone(),
        }
    }
}

impl<T, const R: usize> Ndarr<T, R>
where
    T: Clone + Debug + Default,
{
    pub fn fill(with: T, shape: &[usize; R]) -> Self {
        let data: Vec<T> = vec![with; multiply_list(shape, 1)];
        Ndarr {
            data,
            shape: shape.clone(),
        }
    }
}
