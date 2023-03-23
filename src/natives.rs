use super::*;
use crate::scalars::Scalar;

impl<T, const N: usize> From<[T; N]> for Ndarr<T, 1>
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn from(value: [T; N]) -> Self {
        Ndarr {
            data: value.to_vec(),
            shape: [N],
        }
    }
}

impl<T, const N1: usize, const N2: usize> From<[[T; N1]; N2]> for Ndarr<T, 2>
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn from(value: [[T; N1]; N2]) -> Self {
        let mut data = Vec::with_capacity(N1*N2);
        for row in value.iter() {
            data.extend_from_slice(row);
        }
        Ndarr {
            data,
            shape: [N2, N1],
        }
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize> From<[[[T; N1]; N2]; N3]> for Ndarr<T, 3>
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn from(value: [[[T; N1]; N2]; N3]) -> Self {
        let mut data = Vec::with_capacity(N1*N2*N3);
        for row in value.iter() {
            for column in row.iter() {
                data.extend_from_slice(column)
            }
        }
        Ndarr {
            data,
            shape: [N3, N2, N1],
        }
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize, const N4: usize> From<[[[[T; N1]; N2]; N3]; N4]>
    for Ndarr<T, 4>
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn from(value: [[[[T; N1]; N2]; N3]; N4]) -> Self {
        let mut data = Vec::with_capacity(N1*N2*N3*N4);
        for axis1 in value.iter() {
            for axis2 in axis1.iter() {
                for axis3 in axis2.iter() {
                    data.extend_from_slice(axis3)
                }
            }
        }
        Ndarr {
            data,
            shape: [N4, N3, N2, N1],
        }
    }
}

impl<T> From<Vec<T>> for Ndarr<T, 1>
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn from(value: Vec<T>) -> Self {
        let l = &value.len();
        Ndarr {
            data: value,
            shape: [*l],
        }
    }
}

impl<T> From<std::ops::Range<T>> for Ndarr<T, 1>
where T: Copy + Clone + Debug + Default + Scalar,
std::ops::Range<T>: Iterator,
Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>
{
    fn from(value: std::ops::Range<T>) -> Self {
        let out: Vec<T> = value.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}


impl From<&str> for Ndarr<char, 1> {
    fn from(value: &str) -> Self {
        Ndarr {
            data: value.chars().collect(),
            shape: [value.len()],
        }
    }
}

