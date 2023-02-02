use super::*;
use std::ops::*;

impl<T, const R: usize, const N: usize> Add for Ndarr<T, N, R>
where
    T: Add<Output = T> + Copy + Clone + Debug + Default,
    [T; N]: Default,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        //this is temporary, util we att projection por rank polymorphic operations
        if self.shape != other.shape {
            panic!("Shape missmatch")
        } else {
            self.bimap(other, |x, y| *x + *y)
        }
    }
}
