use super::*;
//impl<T,P> From<C<P>> for C<T> 
//where T: Copy + PartialEq + From<P>,
//P: Copy + PartialEq
//{
   //fn from(value: C<P>) -> Self {
        //C(value.0.into(), value.1.into())     
   //} 
//}

macro_rules!  cast_complex{
    ($T:tt, $P:tt) => {
        impl From<C<$P>> for C<$T>{
        fn from(value: C<$P>) -> Self {
                C(value.0 as $T, value.1 as $T)     
        }}       
    };
}

cast_complex!(u8, u16);
cast_complex!(u8, u32);
cast_complex!(u8, u64);
cast_complex!(u8, u128);
cast_complex!(u8, usize);
cast_complex!(u16, u8);
cast_complex!(u16, u32);
cast_complex!(u16, u64);
cast_complex!(u16, u128);
cast_complex!(u16, usize);
cast_complex!(u32, u8);
cast_complex!(u32, u16);
cast_complex!(u32, u64);
cast_complex!(u32, u128);
cast_complex!(u32, usize);
cast_complex!(u64, u8);
cast_complex!(u64, u16);
cast_complex!(u64, u32);
cast_complex!(u64, u128);
cast_complex!(u64, usize);
cast_complex!(u128, u8);
cast_complex!(u128, u16);
cast_complex!(u128, u32);
cast_complex!(u128, u64);
cast_complex!(usize, u8);
cast_complex!(usize, u16);
cast_complex!(usize, u32);
cast_complex!(usize, u64);
cast_complex!(usize, u128);



cast_complex!(i8, i16);
cast_complex!(i8, i32);
cast_complex!(i8, i64);
cast_complex!(i8, i128);
cast_complex!(i8, isize);
cast_complex!(i16, i8);
cast_complex!(i16, i32);
cast_complex!(i16, i64);
cast_complex!(i16, i128);
cast_complex!(i16, isize);
cast_complex!(i32, i8);
cast_complex!(i32, i16);
cast_complex!(i32, i64);
cast_complex!(i32, i128);
cast_complex!(i32, isize);
cast_complex!(i64, i8);
cast_complex!(i64, i16);
cast_complex!(i64, i32);
cast_complex!(i64, i128);
cast_complex!(i64, isize);
cast_complex!(i128, i8);
cast_complex!(i128, i16);
cast_complex!(i128, i32);
cast_complex!(i128, i64);
cast_complex!(i128, isize);
cast_complex!(isize, i8);
cast_complex!(isize, i16);
cast_complex!(isize, i32);
cast_complex!(isize, i64);
cast_complex!(isize, i128);

cast_complex!(u8, i8);
cast_complex!(u16, i8);
cast_complex!(u16, i16);
cast_complex!(u32, i8);
cast_complex!(u32, i16);
cast_complex!(u32, i32);
cast_complex!(u64, i8);
cast_complex!(u64, i16);
cast_complex!(u64, i32);
cast_complex!(u64, i64);
cast_complex!(u128, i8);
cast_complex!(u128, i16);
cast_complex!(u128, i32);
cast_complex!(u128, i64);
cast_complex!(u128, i128);
cast_complex!(usize, i8);
cast_complex!(usize, i16);
cast_complex!(usize, i32);
cast_complex!(usize, i64);
cast_complex!(usize, i128);
cast_complex!(i8, u8);
cast_complex!(i8, u16);
cast_complex!(i8, u32);
cast_complex!(i8, u64);
cast_complex!(i8, u128);
cast_complex!(i16, u16);
cast_complex!(i16, u32);
cast_complex!(i16, u64);
cast_complex!(i16, u128);
cast_complex!(i32, u32);
cast_complex!(i32, u64);
cast_complex!(i32, u128);
cast_complex!(i64, u64);
cast_complex!(i64, u128);
cast_complex!(i128, u128);

cast_complex!(f32, f64);
cast_complex!(f64, f32);

cast_complex!(i8, f32);
cast_complex!(i8, f64);
cast_complex!(i16, f32);
cast_complex!(i16, f64);
cast_complex!(i32, f32);
cast_complex!(i32, f64);
cast_complex!(i64, f32);
cast_complex!(i64, f64);
cast_complex!(i128, f32);
cast_complex!(i128, f64);
cast_complex!(u8, f32);
cast_complex!(u8, f64);
cast_complex!(u16, f32);
cast_complex!(u16, f64);
cast_complex!(u32, f32);
cast_complex!(u32, f64);
cast_complex!(u64, f32);
cast_complex!(u64, f64);
cast_complex!(u128, f32);
cast_complex!(u128, f64);