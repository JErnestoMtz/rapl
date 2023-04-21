

use super::*;
use std::ops::*;

//-----------------ADD----------------------
macro_rules! complex_add {
    ($LHS:ty, $RHS:ty, $T:tt) => {
        impl<$T: Add<Output = $T> + Copy + PartialEq> Add<$RHS> for $LHS {
            type Output = C<$T>;
            fn add(self, rhs: $RHS) -> Self::Output {
                C(self.0 + rhs.0, self.1 + rhs.1)
            }
        }
    };
}
macro_rules! complex_real_add {
    ($LHS:ty, $T:tt ) => {
        impl<$T: Add<Output = $T> + Copy + PartialEq> Add<$T> for $LHS {
            type Output = C<$T>;
            fn add(self, rhs: $T) -> Self::Output {
                C(self.0 + rhs, self.1)
            }
        }
    };
}

macro_rules! real_complex_add {
    ($T:tt) => {
        impl Add<C<$T>> for $T {
            type Output = C<$T>;
            fn add(self, rhs: C<$T>) -> Self::Output {
                C(self + rhs.0, rhs.1)
            }
        }
    };
}
complex_add!(C<T>, C<T>, T);
complex_add!(&C<T>, &C<T>, T);
complex_add!(C<T>, &C<T>, T);
complex_add!(&C<T>, C<T>, T);

complex_real_add!(C<T>, T);
complex_real_add!(&C<T>, T);

real_complex_add!(u8);
real_complex_add!(u16);
real_complex_add!(u32);
real_complex_add!(u64);
real_complex_add!(u128);
real_complex_add!(i8);
real_complex_add!(i16);
real_complex_add!(i32);
real_complex_add!(i64);
real_complex_add!(i128);
real_complex_add!(f32);
real_complex_add!(f64);
real_complex_add!(usize);
real_complex_add!(isize);

//-----------------SUB----------------------
macro_rules! complex_sub {
    ($LHS:ty, $RHS:ty, $T:tt) => {
        impl<$T: Sub<Output = $T> + Copy + PartialEq> Sub<$RHS> for $LHS {
            type Output = C<$T>;
            fn sub(self, rhs: $RHS) -> Self::Output {
                C(self.0 - rhs.0, self.1 - rhs.1)
            }
        }
    };
}
macro_rules! complex_real_sub {
    ($LHS:ty, $T:tt ) => {
        impl<$T: Sub<Output = $T> + Copy + PartialEq> Sub<$T> for $LHS {
            type Output = C<$T>;
            fn sub(self, rhs: $T) -> Self::Output {
                C(self.0 - rhs, self.1)
            }
        }
    };
}

macro_rules! real_complex_sub {
    ($T:tt) => {
        impl Sub<C<$T>> for $T {
            type Output = C<$T>;
            fn sub(self, rhs: C<$T>) -> Self::Output {
                C(self - rhs.0, -rhs.1)
            }
        }
    };
}

complex_sub!(C<T>, C<T>, T);
complex_sub!(&C<T>, &C<T>, T);
complex_sub!(C<T>, &C<T>, T);
complex_sub!(&C<T>, C<T>, T);

complex_real_sub!(C<T>, T);
complex_real_sub!(&C<T>, T);

real_complex_sub!(i8);
real_complex_sub!(i16);
real_complex_sub!(i32);
real_complex_sub!(i64);
real_complex_sub!(i128);
real_complex_sub!(f32);
real_complex_sub!(f64);
real_complex_sub!(isize);

//-----------------MUl----------------------

macro_rules! complex_mul {
    ($LHS:ty, $RHS:ty, $T:tt ) => {
        impl<$T: Add<Output = $T> + Mul<Output = $T> + Sub<Output = $T> + Copy + PartialEq>
            Mul<$RHS> for $LHS
        {
            type Output = C<$T>;
            fn mul(self, rhs: $RHS) -> Self::Output {
                C(
                    self.0 * rhs.0 - self.1 * rhs.1,
                    self.0 * rhs.1 + self.1 * rhs.0,
                )
            }
        }
    };
}
macro_rules! complex_real_mul {
    ($LHS:ty, $T:tt ) => {
        impl<$T: Add<Output = $T> + Mul<Output = $T> + Sub<Output = $T> + Copy + PartialEq> Mul<$T>
            for $LHS
        {
            type Output = C<$T>;
            fn mul(self, rhs: $T) -> Self::Output {
                C(self.0 * rhs, self.1 * rhs)
            }
        }
    };
}

macro_rules! real_complex_mul {
    ($T:tt) => {
        impl Mul<C<$T>> for $T {
            type Output = C<$T>;
            fn mul(self, rhs: C<$T>) -> Self::Output {
                C(self * rhs.0, self * rhs.1)
            }
        }
    };
}
complex_mul!(C<T>, C<T>, T);
complex_mul!(&C<T>, &C<T>, T);
complex_mul!(C<T>, &C<T>, T);
complex_mul!(&C<T>, C<T>, T);

complex_real_mul!(C<T>, T);
complex_real_mul!(&C<T>, T);

real_complex_mul!(u8);
real_complex_mul!(u16);
real_complex_mul!(u32);
real_complex_mul!(u64);
real_complex_mul!(u128);
real_complex_mul!(i8);
real_complex_mul!(i16);
real_complex_mul!(i32);
real_complex_mul!(i64);
real_complex_mul!(i128);
real_complex_mul!(f32);
real_complex_mul!(f64);
real_complex_mul!(usize);
real_complex_mul!(isize);

//-----------------Div----------------------
macro_rules! complex_div {
    ($LHS:ty, $RHS:ty, $T:tt ) => {
        impl<
                $T: Add<Output = $T>
                    + Mul<Output = $T>
                    + Sub<Output = $T>
                    + Div<Output = T>
                    + Copy
                    + PartialEq,
            > Div<$RHS> for $LHS
        {
            type Output = C<$T>;
            fn div(self, rhs: $RHS) -> Self::Output {
                let den = rhs.0 * rhs.0 + rhs.1 * rhs.1;
                C(
                    (self.0 * rhs.0 + self.1 * rhs.1) / den,
                    (self.1 * rhs.0 - self.0 * rhs.1) / den,
                )
            }
        }
    };
}

macro_rules! complex_real_div {
    ($LHS:ty, $T:tt ) => {
        impl<
                $T: Add<Output = $T>
                    + Mul<Output = $T>
                    + Sub<Output = $T>
                    + Div<Output = T>
                    + Copy
                    + PartialEq,
            > Div<$T> for $LHS
        {
            type Output = C<$T>;
            fn div(self, rhs: $T) -> Self::Output {
                C(self.0 / rhs, self.1 / rhs)
            }
        }
    };
}

macro_rules! real_complex_div {
    ($T:tt ) => {
        impl Div<C<$T>> for $T {
            type Output = C<$T>;
            fn div(self, rhs: C<$T>) -> Self::Output {
                let den = self * self + rhs.1 * rhs.1;
                C((self * rhs.0) / den, (-self * rhs.1) / den)
            }
        }
    };
}
complex_real_div!(C<T>, T);
complex_real_div!(&C<T>, T);

complex_div!(C<T>, C<T>, T);
complex_div!(&C<T>, &C<T>, T);
complex_div!(C<T>, &C<T>, T);
complex_div!(&C<T>, C<T>, T);

real_complex_div!(i8);
real_complex_div!(i16);
real_complex_div!(i32);
real_complex_div!(i64);
real_complex_div!(i128);
real_complex_div!(f32);
real_complex_div!(f64);
real_complex_div!(isize);

//-------------NEG-------------------
macro_rules! complex_neg {
    ($LHS:ty, $T:tt ) => {
        impl<$T: Neg<Output = $T> + Copy + PartialEq> Neg for $LHS {
            type Output = C<$T>;
            fn neg(self) -> Self::Output {
                C(-self.0, -self.1)
            }
        }
    };
}
complex_neg!(C<T>, T);
complex_neg!(&C<T>, T);

//-------------------------------- AddAdding ------------------------------

impl<T: Copy + PartialEq + Add<T, Output = T>> AddAssign<C<T>> for C<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
        self.1 = self.1 + rhs.1;
    }
}
impl<T: Copy + PartialEq + Add<T, Output = T>> AddAssign<T> for C<T> {
    fn add_assign(&mut self, rhs: T) {
        self.0 = self.0 + rhs;
    }
}

//-------------------------------- Sub ------------------------------
impl<T: Copy + PartialEq + Sub<T, Output = T>> SubAssign<C<T>> for C<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
        self.1 = self.1 - rhs.1;
    }
}
impl<T: Copy + PartialEq + Sub<T, Output = T>> SubAssign<T> for C<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.0 = self.0 - rhs;
    }
}

//-------------------------------- Mul ------------------------------
impl<T: Copy + PartialEq + Mul<T, Output = T> + Add<Output = T> + Sub<Output = T>> MulAssign<C<T>>
    for C<T>
{
    fn mul_assign(&mut self, rhs: Self) {
        let out = self.clone() * rhs;
        self.0 = out.0;
        self.1 = out.1;
    }
}
impl<T: Copy + PartialEq + Mul<T, Output = T>> MulAssign<T> for C<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.0 = self.0 * rhs;
        self.1 = self.1 * rhs;
    }
}

//-------------------------------- Div ------------------------------
impl<
        T: Copy + PartialEq + Div<Output = T> + Mul<T, Output = T> + Add<Output = T> + Sub<Output = T>,
    > DivAssign<C<T>> for C<T>
{
    fn div_assign(&mut self, rhs: Self) {
        let out = self.clone() / rhs;
        self.0 = out.0;
        self.1 = out.1;
    }
}
impl<T: Copy + PartialEq + Div<T, Output = T>> DivAssign<T> for C<T> {
    fn div_assign(&mut self, rhs: T) {
        self.0 = self.0 / rhs;
        self.1 = self.1 / rhs;
    }
}
