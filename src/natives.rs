use super::*;
use crate::scalars::Scalar;

impl<T, const N: usize> Into<Ndarr<T, 1>> for [T; N]
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn into(self) -> Ndarr<T, 1> {
        Ndarr {
            data: self.to_vec(),
            shape: [N],
        }
    }
}

impl<T, const N1: usize, const N2: usize> Into<Ndarr<T, 2>> for [[T; N1]; N2]
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn into(self) -> Ndarr<T, 2> {
        let mut data = Vec::new();
        for i in 0..N2 {
            for j in 0..N1 {
                data.push(self[i][j])
            }
        }
        Ndarr {
            data: data,
            shape: [N2, N1],
        }
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize> Into<Ndarr<T, 3>> for [[[T; N1]; N2]; N3]
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn into(self) -> Ndarr<T, 3> {
        let mut data = Vec::new();
        for i in 0..N3 {
            for j in 0..N2 {
                for k in 0..N1 {
                    data.push(self[i][j][k]);
                }
            }
        }
        Ndarr {
            data: data,
            shape: [N3, N2, N1],
        }
    }
}

impl<T, const N1: usize, const N2: usize, const N3: usize, const N4: usize> Into<Ndarr<T, 4>>
    for [[[[T; N1]; N2]; N3]; N4]
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn into(self) -> Ndarr<T, 4> {
        let mut data = Vec::new();
        for i in 0..N4 {
            for j in 0..N3 {
                for k in 0..N2 {
                    for l in 0..N1 {
                        data.push(self[i][j][k][l]);
                    }
                }
            }
        }
        Ndarr {
            data: data,
            shape: [N4, N3, N2, N1],
        }
    }
}

impl<T> Into<Ndarr<T, 1>> for Vec<T>
where
    T: Clone + Debug + Copy + Default + Scalar,
{
    fn into(self) -> Ndarr<T, 1> {
        let l = &self.len();
        Ndarr {
            data: self,
            shape: [*l],
        }
    }
}


impl Into<Ndarr<u32, 1>> for std::ops::Range<u32>
{
    fn into(self) -> Ndarr<u32, 1> {
        let out: Vec<u32> = self.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}

impl Into<Ndarr<u64, 1>> for std::ops::Range<u64>
{
    fn into(self) -> Ndarr<u64, 1> {
        let out: Vec<u64> = self.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}
impl Into<Ndarr<i64, 1>> for std::ops::Range<i64>
{
    fn into(self) -> Ndarr<i64, 1> {
        let out: Vec<i64> = self.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}
impl Into<Ndarr<i32, 1>> for std::ops::Range<i32>
{
    fn into(self) -> Ndarr<i32, 1> {
        let out: Vec<i32> = self.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}
impl Into<Ndarr<usize, 1>> for std::ops::Range<usize>
{
    fn into(self) -> Ndarr<usize, 1> {
        let out: Vec<usize> = self.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}
impl Into<Ndarr<isize, 1>> for std::ops::Range<isize>
{
    fn into(self) -> Ndarr<isize, 1> {
        let out: Vec<isize> = self.collect();
        Ndarr {
            data: out.clone(),
            shape: [out.len()],
        }
    }
}
impl Into<Ndarr<char, 1>> for &str {
    fn into(self) -> Ndarr<char, 1> {
        Ndarr {
            data: self.chars().collect(),
            shape: [self.len()],
        }
    }
}
