use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg};
mod floats;
mod ops;
mod primitives;
mod cast;

pub use crate::complex::primitives::Imag;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub struct C<T: Copy + PartialEq>(pub T, pub T);

impl<T: Copy + PartialEq> C<T> {
    pub fn re(&self) -> T {
        self.0
    }
    pub fn im(&self) -> T {
        self.1
    }
}

impl<T: Copy + PartialEq + Neg<Output = T>> C<T> {
    pub fn conj(&self) -> C<T> {
        C(self.0, -self.1)
    }
}

impl<T: Copy + PartialEq + Add<Output = T> + Mul<Output = T>> C<T> {
    pub fn r_square(&self) -> T {
        self.0 * self.0 + self.1 * self.1
    }
}

impl<T> C<T>
where
    T: Copy + PartialEq + Neg<Output = T> + Div<Output = T> + Mul<Output = T> + Add<Output = T>,
{
    pub fn inv(&self) -> Self {
        let r_sq = self.r_square();
        C(self.0 / r_sq, -self.1 / r_sq)
    }
}



#[cfg(test)]
mod tests {

    use std::f32::consts::PI;

    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(1 + 1.i(), C(1, 1));
        assert_eq!(1.i() + 1, C(1, 1));
        assert_eq!(C(0., 2.) + C(2., 3.), C(2., 5.));
    }

    #[test]
    fn test_sub() {
        assert_eq!(1 - 1.i(), C(1, -1));
        assert_eq!(1.i() - 1, C(-1, 1));
        assert_eq!(C(0., 2.) - C(2., 3.), C(-2., -1.));
    }

    #[test]
    fn test_mul() {
        let c1 = 2 + 3.i();
        let c2 = 4 + 5.i();
        let expected = -7 + 22.i();
        assert_eq!(c1 * c2, expected);
    }

    #[test]
    fn test_division() {
        let c1 = C(2., 3.);
        let c2 = C(4., 5.);
        let expected = C(23. / 41., 2./ 41.);
        assert_eq!(c1 / c2, expected);
    }

    #[test]
    fn test_conj(){
        let a: C<i32> = 2 + 3.i();
        assert!((a * a.conj()).re() == a.r_square())
    }

}
