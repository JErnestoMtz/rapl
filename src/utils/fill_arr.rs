use num_traits::{One, Zero};

use super::helpers::multiply_list;
use super::*;

//// zeros and ones
impl<T, R: Unsigned> Ndarr<T, R>
where
    T: Clone + Debug + Default,
{
    pub fn zeros(shape: &Dim<R>) -> Self
    where
        T: Zero,
    {
        let data: Vec<T> = vec![T::zero(); multiply_list(&shape.shape, 1)];
        Ndarr {
            data,
            dim: shape.clone(),
        }
    }

    pub fn ones(shape: &Dim<R>) -> Self
    where
        T: One,
    {
        let data: Vec<T> = vec![T::one(); multiply_list(&shape.shape, 1)];
        Ndarr {
            data,
            dim: shape.clone(),
        }
    }

    pub fn fill(with: T, shape: &Dim<R>) -> Self {
        let data: Vec<T> = vec![with; multiply_list(&shape.shape, 1)];
        Ndarr {
            data,
            dim: shape.clone(),
        }
    }
}
