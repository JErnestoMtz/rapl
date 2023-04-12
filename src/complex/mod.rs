use num_traits::{Float, MulAddAssign, Num};
use std::{ops::{Add, Div, Mul, Neg, AddAssign, MulAssign}, fmt::Debug};
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

impl<T> C<T>
where
    C<T>: MulAssign + Debug,
    T: Copy + PartialEq + Num
{
    // this seems to to be relatively fast for n < 100, but we should find a better way for larger n's
    pub fn powi(&self, n: i32) -> Self {
        if n == 0{
            return C(T::one(), T::zero())
        }else if n > 0{
            let mut out = self.clone();
            for _ in  1..n{
                out *= self.clone();
            }
            return out
        }else{
            let mut out = self.clone();
            for _ in  1..-n{
                out *= self.clone();
            }
            let out = C(T::one(), T::zero()) / out;
            return  out;
        }
    }
}


#[cfg(test)]
mod tests {
    #![allow(non_upper_case_globals)]
    use std::f32::consts::{PI, FRAC_PI_2, LN_2};
    use std::f64::consts::{FRAC_1_SQRT_2};

    use super::*;

    pub const _0_0: C<f64> = C(0.0, 0.0);
    pub const _1_0: C<f64> = C(1.0, 0.0);
    pub const _0_1: C<f64> = C(0.0, 1.0);
    pub const _n1_0: C<f64> = C(-1.0, 0.0);
    pub const _0_n1: C<f64> = C(-1.0, 0.0);
    pub const _1_1: C<f64> = C(-1.0, 0.0);
    pub const _2_n1: C<f64> = C(-2.0, -1.0);
    pub const unit: C<f64> = C(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    pub const all_z: [C<f64>; 8] = [_0_0, _1_0, _0_1, _n1_0, _0_n1, _1_1, _2_n1, unit];


    fn approx_epsilon(a: C<f64>, b:C<f64>, epsilon: f64) -> bool {
        let approx = (a == b) || (a - b).abs() < epsilon;
        if !approx{
            println!("Error: {:?} != {:?}", a, b)
        }
        approx
    }

    fn approx(a: C<f64>, b:C<f64>) -> bool {
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
        let expected = C(23. / 41., 2./ 41.);
        assert_eq!(c1 / c2, expected);
    }
    #[test]
    fn assing(){
        let mut z = C(0,0);
        z += 2;
        assert_eq!(z, C(2,0));
        z -= 4.i();
        assert_eq!(z, C(2,-4));
        z *= 3;
        assert_eq!(z, C(6,-12));
        z /= C(2,0);
        assert_eq!(z, C(3,-6));
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

    #[test]
    fn abs(){
        assert_eq!(_0_1.abs(), 1.0);
        assert_eq!(_1_0.abs(), 1.0);
        assert_eq!(_n1_0.abs(), 1.0);
        assert_eq!(_0_n1.abs(), 1.0);
        assert_eq!(unit.abs(), 1.0);
    }

    #[test]
    fn sqrt(){
        for n in (0..100).map(f64::from){
            let n2 = n * n;
            assert!(approx(C(n2,0.).sqrt(), C(n,0.)));
            assert!(approx(C(-n2,0.).sqrt(), C(0.,n)));
            assert!(approx(C(-n2,-0.).sqrt(), C(0.0,-n)));
        }
        let z2: C<f64> = 0.25 + 0.0.i();
        assert_eq!(z2.sqrt(), C(0.5,0.));
        for c in all_z{
            assert!(approx(c.conj().sqrt(), c.sqrt().conj()));
            assert!(approx(c.sqrt() * c.sqrt(), c));
            assert!(-std::f64::consts::FRAC_PI_2 <= c.sqrt().arg()&& c.sqrt().arg() <= std::f64::consts::FRAC_PI_2);
        }

    }

    #[test]
    fn powi(){
        let z1 = C(2,0);
        assert_eq!(z1.powi(3), C(8,0));
        let z2 = 2.i();
        assert_eq!(z2.powi(4), C(16,0));
        let z3 = C(3,-5);
        assert_eq!(z3.clone().powi(3), z3.clone() * z3.clone() * z3);
        assert_eq!(_2_n1.powi(2), _2_n1 * _2_n1);
        assert_eq!(C(5,10).powi(0), C(1,0));
        assert_eq!(2.0.i().powi(-2), C(- 1. / 4., 0.));
    }
    #[test]
    fn powf(){
        assert!(approx(_2_n1.powf(2.), _2_n1 * _2_n1));
        assert!(approx(_2_n1.powf(0.), C(1., 0.)));
        assert!(approx(_0_1.powf(4.), C(1.,0.)))
    }
    #[test]
    fn powc(){
        assert!(approx(_2_n1.powc(C(2.,0.)), _2_n1 * _2_n1));
        assert!(approx(_2_n1.powc(C(0., 0.)), C(1., 0.)));
        //form python 
        //>>> z = 2.0 + 0.5j
        //>>> z**z
        //(2.4767939208048335+2.8290270856372506j)
        let z: C<f64> = 2.0 + 0.5.i();
        assert!(approx(z.powc(z.clone()),C(2.4767939208048335,2.8290270856372506)))
    }
    
}
 
