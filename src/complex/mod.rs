use num_traits::{Num, Signed};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg},
};
mod floats;
mod ops;
mod primitives;

pub use crate::complex::primitives::Imag;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub struct C<T>(pub T, pub T);

impl<T: Display + Signed> Display for C<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sign = if self.1.is_negative() { "" } else { "+" };
        match f.precision() {
            Some(p) => write!(f, "{:.p$}{sign}{:.p$}i", self.0, self.1),
            None => write!(f, "{}{sign}{}i", self.0, self.1),
        }
    }
}

impl<T: Copy> C<T> {
    pub fn re(&self) -> T {
        self.0
    }
    pub fn im(&self) -> T {
        self.1
    }
}

impl<T: Copy + Neg<Output = T>> C<T> {
    pub fn conj(&self) -> C<T> {
        C(self.0, -self.1)
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T>> C<T> {
    pub fn r_square(&self) -> T {
        self.0 * self.0 + self.1 * self.1
    }
}

impl<T: Num> From<T> for C<T> {
    fn from(value: T) -> Self {
        C(value, T::zero())
    }
}

impl<T> C<T>
where
    T: Copy + Neg<Output = T> + Div<Output = T> + Mul<Output = T> + Add<Output = T>,
{
    pub fn inv(&self) -> Self {
        let r_sq = self.r_square();
        C(self.0 / r_sq, -self.1 / r_sq)
    }
}

impl<T: Copy + Num> C<T> {
    pub fn powi(&self, n: i32) -> Self {
        let one = C(T::one(), T::zero());
        let pow = (0..n.unsigned_abs()).fold(one, |acc, _| acc * *self);
        if n < 0 {
            one / pow
        } else {
            pow
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_upper_case_globals)]
    use std::f64::consts::FRAC_1_SQRT_2;

    use super::*;

    pub const _0_0: C<f64> = C(0.0, 0.0);
    pub const _1_0: C<f64> = C(1.0, 0.0);
    pub const _0_1: C<f64> = C(0.0, 1.0);
    pub const _n1_0: C<f64> = C(-1.0, 0.0);
    pub const _0_n1: C<f64> = C(0.0, -1.0);
    pub const _1_1: C<f64> = C(1.0, 1.0);
    pub const _2_n1: C<f64> = C(-2.0, -1.0);
    pub const unit: C<f64> = C(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    pub const all_z: [C<f64>; 8] = [_0_0, _1_0, _0_1, _n1_0, _0_n1, _1_1, _2_n1, unit];

    fn approx_epsilon(a: C<f64>, b: C<f64>, epsilon: f64) -> bool {
        let approx = (a == b) || (a - b).abs() < epsilon;
        if !approx {
            println!("Error: {:?} != {:?}", a, b)
        }
        approx
    }

    fn approx(a: C<f64>, b: C<f64>) -> bool {
        approx_epsilon(a, b, 1e-10)
    }
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
        let expected = C(23. / 41., 2. / 41.);
        assert_eq!(c1 / c2, expected);
    }
    #[test]
    fn assign() {
        let mut z = C(0, 0);
        z += 2;
        assert_eq!(z, C(2, 0));
        z -= 4.i();
        assert_eq!(z, C(2, -4));
        z *= 3;
        assert_eq!(z, C(6, -12));
        z /= C(2, 0);
        assert_eq!(z, C(3, -6));
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
        assert_eq!(a_complex, C(42, 0));

        let a: f32 = 42.0;
        let a_complex = C::from(a);
        assert_eq!(a_complex, C(42.0, 0.0));

        let a: i32 = -42;
        let a_complex = C::from(a);
        assert_eq!(a_complex, C(-42, 0));
    }

    #[test]
    fn ln() {
        // Use f64 + approx: complex ln of axis values is not bit-exact on all platforms.
        let a = C(0.0_f64, 1.0);
        assert!(approx(a.ln(), C(0.0, std::f64::consts::FRAC_PI_2)));

        let a = C(2.0_f64, 0.0);
        assert!(approx(a.ln(), C(std::f64::consts::LN_2, 0.0)));

        let a = C(-1.0_f64, 0.0);
        assert!(approx(a.ln(), C(0.0, std::f64::consts::PI)));
    }

    #[test]
    fn abs() {
        assert_eq!(_0_1.abs(), 1.0);
        assert_eq!(_1_0.abs(), 1.0);
        assert_eq!(_n1_0.abs(), 1.0);
        assert_eq!(_0_n1.abs(), 1.0);
        assert_eq!(unit.abs(), 1.0);
    }

    #[test]
    fn sqrt() {
        for n in (0..100).map(f64::from) {
            let n2 = n * n;
            assert!(approx(C(n2, 0.).sqrt(), C(n, 0.)));
            assert!(approx(C(-n2, 0.).sqrt(), C(0., n)));
            assert!(approx(C(-n2, -0.).sqrt(), C(0.0, -n)));
        }
        let z2: C<f64> = 0.25 + 0.0.i();
        assert_eq!(z2.sqrt(), C(0.5, 0.));
        for c in all_z {
            assert!(approx(c.conj().sqrt(), c.sqrt().conj()));
            assert!(approx(c.sqrt() * c.sqrt(), c));
            assert!(
                -std::f64::consts::FRAC_PI_2 <= c.sqrt().arg()
                    && c.sqrt().arg() <= std::f64::consts::FRAC_PI_2
            );
        }
    }

    #[test]
    fn powi() {
        let z1 = C(2, 0);
        assert_eq!(z1.powi(3), C(8, 0));
        let z2 = 2.i();
        assert_eq!(z2.powi(4), C(16, 0));
        let z3 = C(3, -5);
        assert_eq!(z3.clone().powi(3), z3 * z3 * z3);
        assert_eq!(_2_n1.powi(2), _2_n1 * _2_n1);
        assert_eq!(C(5, 10).powi(0), C(1, 0));
        assert_eq!(2.0.i().powi(-2), C(-1. / 4., 0.));
    }
    #[test]
    fn powf() {
        assert!(approx(_2_n1.powf(2.), _2_n1 * _2_n1));
        assert!(approx(_2_n1.powf(0.), C(1., 0.)));
        assert!(approx(_0_1.powf(4.), C(1., 0.)))
    }
    #[test]
    fn powc() {
        assert!(approx(_2_n1.powc(C(2., 0.)), _2_n1 * _2_n1));
        assert!(approx(_2_n1.powc(C(0., 0.)), C(1., 0.)));
        // Reference value from python.
        let z: C<f64> = 2.0 + 0.5.i();
        assert!(approx(z.powc(z), C(2.4767939208048335, 2.8290270856372506)))
    }
}
