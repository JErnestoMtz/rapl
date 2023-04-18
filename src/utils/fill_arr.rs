
use num_traits::{Zero, One};

use super::*;
use super::helpers::multiply_list;

//// zeros and ones
impl<T, const R: usize> Ndarr<T, R>
where
    T: Clone + Debug + Default, 
{
    pub fn zeros(shape: &[usize; R]) -> Self 
        where T: Zero 
    {
        let data: Vec<T> = vec![T::zero(); multiply_list(shape, 1)];
        Ndarr {
            data,
            shape: shape.clone(),
        }
    }

    pub fn ones(shape: &[usize; R]) -> Self 
        where T: One 
    {
        let data: Vec<T> = vec![T::one(); multiply_list(shape, 1)];
        Ndarr {
            data,
            shape: shape.clone(),
        }
    }

    pub fn fill(with: T, shape: &[usize; R]) -> Self {
        let data: Vec<T> = vec![with; multiply_list(shape, 1)];
        Ndarr {
            data,
            shape: shape.clone(),
        }
    }

}


