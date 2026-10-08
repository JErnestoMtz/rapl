use super::*;
use std::ops::*;

macro_rules! ndarr_op {
    ($Ty1:ty, $Ty2:ty, $Trait:tt, $F:tt, $Op:tt) => {
        impl<T1, T2, T3, R1: Rank, R2: Rank, B1, B2> $Trait<$Ty2> for $Ty1
        where
            R1: BroadcastRank<R2>,
            T1: Clone + $Trait<T2, Output = T3>,
            T2: Clone,
            B1: Buffer<T1>,
            B2: Buffer<T2>,
        {
            type Output = Ndarr<T3, Broadcasted<R1, R2>>;
            fn $F(self, rhs: $Ty2) -> Self::Output {
                self.zip_with(&rhs, |x, y| x.clone() $Op y.clone()).unwrap()
            }
        }
    };
}

macro_rules! ndarr_ops {
    ($Trait:tt, $F:tt, $Op:tt) => {
        ndarr_op!(Ndarr<T1, R1, B1>, Ndarr<T2, R2, B2>, $Trait, $F, $Op);
        ndarr_op!(Ndarr<T1, R1, B1>, &Ndarr<T2, R2, B2>, $Trait, $F, $Op);
        ndarr_op!(&Ndarr<T1, R1, B1>, Ndarr<T2, R2, B2>, $Trait, $F, $Op);
        ndarr_op!(&Ndarr<T1, R1, B1>, &Ndarr<T2, R2, B2>, $Trait, $F, $Op);
    };
}

ndarr_ops!(Add, add, +);
ndarr_ops!(Sub, sub, -);
ndarr_ops!(Mul, mul, *);
ndarr_ops!(Div, div, /);
ndarr_ops!(Rem, rem, %);

macro_rules! scalar_op {
    ($Op:tt, $f_name:tt, $f:tt) => {
        impl<T, L, P, R: Rank, B: Buffer<T>> $Op<P> for Ndarr<T, R, B>
        where
            T: Clone + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L, R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map(|x| x.clone() $f other)
            }
        }
        impl<T, L, P, R: Rank, B: Buffer<T>> $Op<P> for &Ndarr<T, R, B>
        where
            T: Clone + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L, R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map(|x| x.clone() $f other)
            }
        }
    };
}

scalar_op!(Add, add, +);
scalar_op!(Sub, sub, -);
scalar_op!(Mul, mul, *);
scalar_op!(Div, div, /);
scalar_op!(Rem, rem, %);

// Spelled per primitive type: a blanket impl puts an uncovered type
// parameter in the self position of a foreign trait (E0210).
macro_rules! scalar_op2 {
    ($Op:tt, $f_name:tt, $f:tt, $t:ty) => {
        impl<T, R: Rank, B: Buffer<T>> $Op<Ndarr<T, R, B>> for $t
        where
            T: Clone + $Op<$t, Output = T>,
        {
            type Output = Ndarr<T, R>;
            fn $f_name(self, rhs: Ndarr<T, R, B>) -> Self::Output {
                rhs.map(|x| x.clone() $f self)
            }
        }
        impl<T, R: Rank, B: Buffer<T>> $Op<&Ndarr<T, R, B>> for $t
        where
            T: Clone + $Op<$t, Output = T>,
        {
            type Output = Ndarr<T, R>;
            fn $f_name(self, rhs: &Ndarr<T, R, B>) -> Self::Output {
                rhs.map(|x| x.clone() $f self)
            }
        }
    };
}
macro_rules! scalar_to_ndarr {
    ($($t:ty),+ $(,)?) => {
        $(
            scalar_op2!(Add, add, +, $t);
            scalar_op2!(Sub, sub, -, $t);
            scalar_op2!(Mul, mul, *, $t);
            scalar_op2!(Div, div, /, $t);
            scalar_op2!(Rem, rem, %, $t);
        )+
    };
}

scalar_to_ndarr!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, char);

impl<T, R: Rank, B: Buffer<T>> Neg for Ndarr<T, R, B>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Ndarr<T, R>;
    fn neg(self) -> Self::Output {
        self.map(|x| -x.clone())
    }
}

impl<T, R: Rank, B: Buffer<T>> Neg for &Ndarr<T, R, B>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Ndarr<T, R>;
    fn neg(self) -> Self::Output {
        self.map(|x| -x.clone())
    }
}

// Array right-hand sides update in place via `zip_with_in_place`, broadcasting to
// the left-hand shape; scalars via `map_in_place`. Operators cannot return
// `Result`, so a right-hand side that does not broadcast panics.
macro_rules! assign_op {
    ($Trait:tt, $f_name:tt, $Op:tt, $f:tt) => {
        impl<T, R: Rank, R2: Rank, B: BufferMut<T>, B2: Buffer<T>> $Trait<&Ndarr<T, R2, B2>>
            for Ndarr<T, R, B>
        where
            T: Clone + $Op<Output = T>,
        {
            fn $f_name(&mut self, other: &Ndarr<T, R2, B2>) {
                self.zip_with_in_place(other, |x, y| x.clone() $f y.clone())
                    .expect("the right-hand side must broadcast to the left-hand shape")
            }
        }

        impl<T, P, R: Rank, B: BufferMut<T>> $Trait<P> for Ndarr<T, R, B>
        where
            T: Clone + $Op<P, Output = T>,
            P: Scalar + Copy,
        {
            fn $f_name(&mut self, scalar: P) {
                self.map_in_place(|x| x.clone() $f scalar)
            }
        }
    };
}

assign_op!(AddAssign, add_assign, Add, +);
assign_op!(SubAssign, sub_assign, Sub, -);
assign_op!(MulAssign, mul_assign, Mul, *);
assign_op!(DivAssign, div_assign, Div, /);
assign_op!(RemAssign, rem_assign, Rem, %);

#[cfg(test)]
mod test_arithmetics {
    use super::*;
    #[test]
    fn test_basic() {
        let arr1 = Ndarr::from([1, 2, 3]);
        let arr2 = Ndarr::from([1, 1, 1]);
        let arr3 = Ndarr::from([2, 2, 2]);
        assert_eq!(&arr1 - &arr2, Ndarr::from([0, 1, 2]));
        assert_eq!(&arr1 + arr2, Ndarr::from([2, 3, 4]));
        assert_eq!(arr1 * arr3, Ndarr::from([2, 4, 6]));
    }

    #[test]
    fn test_single_broadcast() {
        let arr1 = Ndarr::from([1, 2]);
        let arr2 = Ndarr::from([[1, 2], [3, 4]]);
        assert_eq!(&arr1 + &arr2, Ndarr::from([[2, 4], [4, 6]]));
    }

    #[test]
    fn test_cobroadcast() {
        let arr1 = Ndarr::from([[1, 2, 3]]);
        assert_eq!(
            &arr1 + arr1.t(),
            Ndarr::from([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
        );
    }

    #[test]
    fn test_sclalat() {
        let arr = Ndarr::from([0.1, 0.2, 0.3]);
        let arr_scalar: Ndarr<f64, _> = &arr * 2.0;
        let scalar_arr = 2.0 * arr;
        assert_eq!(arr_scalar, scalar_arr)
    }
}
