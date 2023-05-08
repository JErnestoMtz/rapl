use num_traits::{One, Zero};

use super::helpers::multiply_list;
use super::*;

//// zeros and ones
impl<T, R: Unsigned> Ndarr<T, R>
where
    T: Clone + Debug + Default,
{
    pub fn zeros<D: Into<Dim<R>>>(shape: D) -> Self
    where
        T: Zero,
    {
        let shape = shape.into();
        let data: Vec<T> = vec![T::zero(); multiply_list(&shape.shape, 1)];
        Ndarr {
            data,
            dim: shape,
        }
    }

    pub fn ones<D: Into<Dim<R>>>(shape: D) -> Self
    where
        T: One,
    {

        let shape = shape.into();
        let data: Vec<T> = vec![T::one(); multiply_list(&shape.shape, 1)];
        Ndarr {
            data,
            dim: shape,
        }
    }

    pub fn fill<D: Into<Dim<R>>>(with: T, shape: D) -> Self {
        let shape = shape.into();
        let data: Vec<T> = vec![with; multiply_list(&shape.shape, 1)];
        Ndarr {
            data,
            dim: shape,
        }
    }
}
