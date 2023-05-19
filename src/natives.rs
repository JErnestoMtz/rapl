use super::*;
use crate::scalars::Scalar;
use typenum::{U1, U2, U3, U4};

impl<T, const N: usize> From<[T; N]> for Ndarr<T, U1>
where
    T: Clone + Debug + Scalar,
{
    fn from(value: [T; N]) -> Self {
        Ndarr {
            data: value.to_vec(),
            dim: Dim::<U1>::new(&[N]).unwrap(),
        }
    }
}

impl<T, const N1: usize, const N2: usize> From<[[T; N1]; N2]> for Ndarr<T, U2>
where
    T: Clone + Debug + Scalar,
{
    fn from(value: [[T; N1]; N2]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2);
        for row in value.iter() {
            data.extend_from_slice(row);
        }
        Ndarr {
            data,
            dim: Dim::new(&[N2, N1]).unwrap(),
        }
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize> From<[[[T; N1]; N2]; N3]>
    for Ndarr<T, U3>
where
    T: Clone + Debug  + Scalar,
{
    fn from(value: [[[T; N1]; N2]; N3]) -> Self {
        let mut data = Vec::with_capacity(N1 * N2 * N3);
        for row in value.iter() {
            for column in row.iter() {
                data.extend_from_slice(column)
            }
        }
        Ndarr {
            data,
            dim: Dim::new(&[N3, N2, N1]).unwrap(),
        }
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize, const N4: usize>
    From<[[[[T; N1]; N2]; N3]; N4]> for Ndarr<T, U4>
where
    T: Clone + Debug  + Scalar,
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
        Ndarr {
            data,
            dim: Dim::new(&[N4, N3, N2, N1]).unwrap(),
        }
    }
}

impl<T> From<Vec<T>> for Ndarr<T, U1>
where
    T: Clone + Debug + Scalar,
{
    fn from(value: Vec<T>) -> Self {
        let l = &value.len();
        Ndarr {
            data: value,
            dim: Dim::new(&[*l]).unwrap(),
        }
    }
}

impl<T> From<std::ops::Range<T>> for Ndarr<T, U1>
where
    T: Clone + Debug  + Scalar,
    std::ops::Range<T>: Iterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    fn from(value: std::ops::Range<T>) -> Self {
        let out: Vec<T> = value.collect();
        Ndarr {
            data: out.clone(),
            dim: Dim::new(&[out.len()]).unwrap(),
        }
    }
}

impl From<&str> for Ndarr<char, U1> {
    fn from(value: &str) -> Self {
        Ndarr {
            data: value.chars().collect(),
            dim: Dim::new(&[value.len()]).unwrap(),
        }
    }
}
