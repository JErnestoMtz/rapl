use super::*;
use std::ops::*;
use typenum::{Maximum, Unsigned};

macro_rules!  ndarr_op{
    ($Ty1:ty, $Ty2:ty, $Trait:tt, $F:tt, $Op:tt) => {

        impl <T1, T2, T3, R1: Unsigned, R2: Unsigned> $Trait<$Ty2> for $Ty1
        where
            R1: Max<R2>,
            <R1 as Max<R2>>::Output: Unsigned,
            T1: Clone + Debug + Default + $Trait<T2, Output = T3>,
            T2: Clone + Debug + Default,
            T3: Clone + Debug + Default,
        {
            type Output = Ndarr<T3,Maximum<R1,R2>>;
            fn $F(self, rhs: $Ty2) -> Self::Output {
                self.poly_dyadic(&rhs, |x,y| x $Op y).unwrap()
            }
        }
    };
}
//--------------------------------- Add --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Add, add, +);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Add, add, +);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Add, add, +);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Add, add, +);

//--------------------------------- Sub --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Sub, sub, -);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Sub, sub, -);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Sub, sub, -);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Sub, sub, -);

//--------------------------------- Mul --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Mul, mul, *);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Mul, mul, *);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Mul, mul, *);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Mul, mul, *);

//--------------------------------- Div --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Div, div, /);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Div, div, /);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Div, div, /);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Div, div, /);

//--------------------------------- Rem --------------------------------------
ndarr_op!(Ndarr<T1,R1>,   Ndarr<T2,R2>, Rem, rem, %);
ndarr_op!(Ndarr<T1,R1>,  &Ndarr<T2,R2>, Rem, rem, %);
ndarr_op!(&Ndarr<T1,R1>,  Ndarr<T2,R2>, Rem, rem, %);
ndarr_op!(&Ndarr<T1,R1>, &Ndarr<T2,R2>, Rem, rem, %);

//////////////////////////////// Scalars ////////////////////////////////////
macro_rules! scalar_op {
    ($Op:tt, $f_name:tt, $f:tt) => {
        impl<L,P, T, R: Unsigned> $Op<P> for Ndarr<T, R>
        where
            L: Clone + Debug + Default,
            T: Clone + Debug + Default + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L,R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map(|x| x.clone() $f other)
            }
        }
        impl<L,P, T, R: Unsigned> $Op<P> for &Ndarr<T, R>
        where
            L: Clone + Debug + Default,
            T: Clone + Debug + Default + $Op<P, Output = L>,
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

macro_rules! scalar_op2 {
    ($Op:tt, $f_name:tt, $f:tt, $t:ty) => {
        impl<T, R: Unsigned> $Op<Ndarr<T,R>> for $t
            where T: Clone + Debug + Default + $Op<$t,Output = T>,
        {
            type Output = Ndarr<T,R>;
            fn $f_name(self, rhs: Ndarr<T,R>) -> Self::Output {
                rhs.map(|x| x.clone() $f self)
            }
        }
        impl<T, R: Unsigned> $Op<&Ndarr<T,R>> for $t
            where T: Clone + Debug + Default + $Op<$t,Output = T>,
        {
            type Output = Ndarr<T,R>;
            fn $f_name(self, rhs: &Ndarr<T,R>) -> Self::Output {
                rhs.map(|x| x.clone() $f self)
            }
        }
    }
}
macro_rules! scalar_to_ndarr {
    ($t:ty) => {
        scalar_op2!(Add, add, +, $t);
        scalar_op2!(Sub, sub, -, $t);
        scalar_op2!(Mul, mul, *, $t);
        scalar_op2!(Div, div, /, $t);
        scalar_op2!(Rem, rem, %, $t);
    };
}

scalar_to_ndarr!(u8);
scalar_to_ndarr!(u16);
scalar_to_ndarr!(u32);
scalar_to_ndarr!(u64);
scalar_to_ndarr!(u128);
scalar_to_ndarr!(i8);
scalar_to_ndarr!(i16);
scalar_to_ndarr!(i32);
scalar_to_ndarr!(i64);
scalar_to_ndarr!(i128);

scalar_to_ndarr!(f32);
scalar_to_ndarr!(f64);

scalar_to_ndarr!(char);

//////////////////////////////////////////// Neg /////////////////////////////////////////////

impl<T, R: Unsigned> Neg for Ndarr<T, R>
where
    T: Neg<Output = T> + Clone + Debug + Default + Copy,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}

impl<T, R: Unsigned> Neg for &Ndarr<T, R>
where
    T: Neg<Output = T> + Clone + Debug + Default + Copy,
{
    type Output = Ndarr<T, R>;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}

//////////////////////////////////////////// AddAssing /////////////////////////////////////////////

impl<P, T, R: Unsigned> AddAssign<&P> for Ndarr<T, R>
where
    T: Add<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn add_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.dim), |x, y| x + y)
    }
}

////////////////////////////////////////////  SubAssing /////////////////////////////////////////////

impl<P, T, R: Unsigned> SubAssign<&P> for Ndarr<T, R>
where
    T: Sub<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn sub_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.dim), |x, y| x - y)
    }
}

////////////////////////////////////////////  MulAssing /////////////////////////////////////////////

impl<P, T, R: Unsigned> MulAssign<&P> for Ndarr<T, R>
where
    T: Mul<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn mul_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.dim), |x, y| x * y)
    }
}

////////////////////////////////////////////  DivAssing /////////////////////////////////////////////

impl<P, T, R: Unsigned> DivAssign<&P> for Ndarr<T, R>
where
    T: Div<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn div_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.dim), |x, y| x / y)
    }
}

////////////////////////////////////////////  RemAssing /////////////////////////////////////////////

impl<P, T, R: Unsigned> RemAssign<&P> for Ndarr<T, R>
where
    T: Rem<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn rem_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.dim), |x, y| x % y)
    }
}
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
