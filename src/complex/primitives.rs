use super::*;
use num_traits::{Num, identities::zero};

pub trait Imag<T: Copy + PartialEq> {
    fn i(&self) -> C<T>;
}

impl<T: Num + Copy> Imag<T> for T {
    fn i(&self) -> C<T> {
        C(zero(), *self)
    }
}
