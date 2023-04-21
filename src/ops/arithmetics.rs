use super::dyadic::*;
use super::*;
use std::ops::*;

macro_rules!  ndarr_op{
    ($Ty1:ty, $Ty2:ty, $Trait:tt, $F:tt, $Op:tt) => {

        impl <T1, T2, T3, const R1: usize, const R2: usize> $Trait<$Ty2> for $Ty1
        where
            T1: Clone + Debug + Default + $Trait<T2, Output = T3>,
            T2: Clone + Debug + Default,
            T3: Clone + Debug + Default,
            [usize; const_max(R2, R1)]: Sized,
            [usize; const_max(R1, R2)]: Sized,
        {
            type Output = Ndarr<T3,{const_max(R1,  R2)}>;
            fn $F(self, rhs: $Ty2) -> Self::Output {
                poly_diatic(&self, &rhs, |x,y| x $Op y).unwrap()
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
        impl<L,P, T, const R: usize> $Op<P> for Ndarr<T, R>
        where
            L: Clone + Debug + Default,
            T: Clone + Debug + Default + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L,R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map_types(|x| x.clone() $f other)
            }
        }
        impl<L,P, T, const R: usize> $Op<P> for &Ndarr<T, R>
        where
            L: Clone + Debug + Default,
            T: Clone + Debug + Default + $Op<P, Output = L>,
            P: Scalar + Copy,
        {
            type Output = Ndarr<L, R>;
            fn $f_name(self, other: P) -> Self::Output {
                self.map_types(|x| x.clone() $f other)
            }
        }
    };
}

scalar_op!(Add, add, +);
scalar_op!(Sub, sub, -);
scalar_op!(Mul, mul, *);
scalar_op!(Div, div, /);
scalar_op!(Rem, rem, %);

//////////////////////////////////////////// Neg /////////////////////////////////////////////

impl<T, const R: usize> Neg for Ndarr<T, R>
where
    T: Neg<Output = T> + Clone + Debug + Default + Copy,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}

impl<T, const R: usize> Neg for &Ndarr<T, R>
where
    T: Neg<Output = T> + Clone + Debug + Default + Copy,
{
    type Output = Ndarr<T, R>;
    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}

//////////////////////////////////////////// AddAssing /////////////////////////////////////////////

impl<P, T, const R: usize> AddAssign<&P> for Ndarr<T, R>
where
    T: Add<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn add_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x, y| x + y)
    }
}

////////////////////////////////////////////  SubAssing /////////////////////////////////////////////

impl<P, T, const R: usize> SubAssign<&P> for Ndarr<T, R>
where
    T: Sub<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn sub_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x, y| x - y)
    }
}

////////////////////////////////////////////  MulAssing /////////////////////////////////////////////

impl<P, T, const R: usize> MulAssign<&P> for Ndarr<T, R>
where
    T: Mul<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn mul_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x, y| x * y)
    }
}

////////////////////////////////////////////  DivAssing /////////////////////////////////////////////

impl<P, T, const R: usize> DivAssign<&P> for Ndarr<T, R>
where
    T: Div<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn div_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x, y| x / y)
    }
}

////////////////////////////////////////////  RemAssing /////////////////////////////////////////////

impl<P, T, const R: usize> RemAssign<&P> for Ndarr<T, R>
where
    T: Rem<Output = T> + Clone + Debug + Default,
    P: IntoNdarr<T, R> + Clone,
{
    fn rem_assign(&mut self, other: &P) {
        self.bimap_in_place(&other.into_ndarr(&self.shape), |x, y| x % y)
    }
}
