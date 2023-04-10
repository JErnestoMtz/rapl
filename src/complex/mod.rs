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

    use std::f32::consts::{PI, FRAC_PI_2, LN_2};

    use super::*;

    #[test]
    fn add() {
        assert_eq!(1 + 1.i(), C(1, 1));
        assert_eq!(1.i() + 1, C(1, 1));
        assert_eq!(C(0., 2.) + C(2., 3.), C(2., 5.));
    }

    #[test]
    fn sub() {
        assert_eq!(1 - 1.i(), C(1, -1));
        assert_eq!(1.i() - 1, C(-1, 1));
        assert_eq!(C(0., 2.) - C(2., 3.), C(-2., -1.));
    }

    #[test]
    fn mul() {
        let c1 = 2 + 3.i();
        let c2 = 4 + 5.i();
        let expected = -7 + 22.i();
        assert_eq!(c1 * c2, expected);
    }

    #[test]
    fn division() {
        let c1 = C(2., 3.);
        let c2 = C(4., 5.);
        let expected = C(23. / 41., 2./ 41.);
        assert_eq!(c1 / c2, expected);
    }

    #[test]
    fn conj() {
        let a: C<i32> = 2 + 3.i();
        assert!((a * a.conj()).re() == a.r_square())
    }

    #[test]
    fn from_num() {
        let a: u8 = 42;
        let a_complex = C::from(a);
        assert_eq!(a_complex, C(42,0));

        let a: f32 = 42.0;
        let a_complex = C::from(a);
        assert_eq!(a_complex, C(42.0,0.0));

        let a: i32 = -42;
        let a_complex = C::from(a);
        assert_eq!(a_complex, C(-42,0));
    }

    #[test]
    fn ln() {
        let a = C(0.0, 1.0);
        assert_eq!(a.ln(), C(0.0, FRAC_PI_2));

        let a = C(2.0, 0.0);
        assert_eq!(a.ln(), C(LN_2, 0.0));

        let a = C(-1.0, 0.0);
        assert_eq!(a.ln(), C(0.0, PI));
    }
}
 