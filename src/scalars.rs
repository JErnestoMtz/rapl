/// Marker for element-like types, keeping blanket scalar operator impls
/// coherent next to the array impls.
pub trait Scalar {}

macro_rules! impl_scalar {
    ($($type:ty),+ $(,)?) => {
        $(impl Scalar for $type {})+
    };
}

impl_scalar!(f64, f32, i128, i64, i32, i16, i8, isize, u128, u64, u32, u16, u8, usize, char);

impl<T: Scalar> Scalar for &T {}
