use super::*;
use crate::scalars::Scalar;
use typenum::{U1, U2, U3, U4};

// `T: Scalar` on the array literal impls is a deliberate disambiguator:
// without it a nested literal like `[[1, 2], [3, 4]]` would match both the
// rank-2 impl (T = i32) and the rank-1 impl (T = [i32; 2]) and fail inference.
// Cost: non-`Scalar` elements (e.g. `String`) must use `Ndarr::new`.
impl<T, const N: usize> From<[T; N]> for Ndarr<T, U1>
where
    T: Clone + Scalar,
{
    fn from(value: [T; N]) -> Self {
        Ndarr::contiguous(value.to_vec(), Dim::from([N]))
    }
}

impl<T, const N1: usize, const N2: usize> From<[[T; N1]; N2]> for Ndarr<T, U2>
where
    T: Clone + Scalar,
{
    fn from(value: [[T; N1]; N2]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2);
        for row in value.iter() {
            data.extend_from_slice(row);
        }
        Ndarr::contiguous(data, Dim::from([N2, N1]))
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize> From<[[[T; N1]; N2]; N3]>
    for Ndarr<T, U3>
where
    T: Clone + Scalar,
{
    fn from(value: [[[T; N1]; N2]; N3]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2 * N3);
        for row in value.iter() {
            for column in row.iter() {
                data.extend_from_slice(column)
            }
        }
        Ndarr::contiguous(data, Dim::from([N3, N2, N1]))
    }
}

// Top rank: the `Scalar` bound keeps the family uniform — a rank-5 literal is
// an error instead of silently building a rank-4 array of array elements.
impl<T, const N1: usize, const N2: usize, const N3: usize, const N4: usize>
    From<[[[[T; N1]; N2]; N3]; N4]> for Ndarr<T, U4>
where
    T: Clone + Scalar,
{
    fn from(value: [[[[T; N1]; N2]; N3]; N4]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2 * N3 * N4);
        for axis1 in value.iter() {
            for axis2 in axis1.iter() {
                for axis3 in axis2.iter() {
                    data.extend_from_slice(axis3)
                }
            }
        }
        Ndarr::contiguous(data, Dim::from([N4, N3, N2, N1]))
    }
}

// No nesting ambiguity for `Vec`/`Range`, so no `Scalar` bound: a
// `Vec<String>` converts directly.
impl<T> From<Vec<T>> for Ndarr<T, U1> {
    fn from(value: Vec<T>) -> Self {
        let l = value.len();
        Ndarr::contiguous(value, Dim::from([l]))
    }
}

impl<T> From<std::ops::Range<T>> for Ndarr<T, U1>
where
    std::ops::Range<T>: Iterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    fn from(value: std::ops::Range<T>) -> Self {
        let out: Vec<T> = value.collect();
        let len = out.len();
        Ndarr::contiguous(out, Dim::from([len]))
    }
}

impl From<&str> for Ndarr<char, U1> {
    fn from(value: &str) -> Self {
        let chars: Vec<char> = value.chars().collect();
        let len = chars.len();
        Ndarr::contiguous(chars, Dim::from([len]))
    }
}
