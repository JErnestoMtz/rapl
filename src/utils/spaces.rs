use std::ops::{Add, Div, Mul, Sub};

use super::{Ndarr, U1, U2};
use num_traits::{Float, Pow, Unsigned};
use std::fmt::Debug;

impl<T: Debug + Clone + Add<Output = T> + Div<Output = T> + Sub<Output = T>> Ndarr<T, U1> {
    ///Return evenly spaced numbers over a specified interval.
    pub fn linspace(start: T, end: T, n: u16) -> Self
    where
        u16: TryInto<T>,
        <u16 as TryInto<T>>::Error: Debug,
    {
        let segments: T = (n - 1).try_into().expect("n too large, max value 2^16");

        let dx = (end - start.clone()) / segments;

        let mut data: Vec<T> = vec![start; n as usize];

        for i in 1..n as usize {
            data[i] = data[i - 1].clone() + dx.clone();
        }
        Ndarr::new(&data, [data.len()]).unwrap()
    }
    ///Return an Ndarr numbers spaced evenly on a log scale.
    pub fn logspace(start: T, end: T, base: T, n: u16) -> Self
    where
        u16: TryInto<T>,
        <u16 as TryInto<T>>::Error: Debug,
        T: Pow<T, Output = T> + Mul<Output = T> + Copy,
    {
        let segments: T = (n - 1).try_into().expect("n too large, max value 2^16");
        let step = (end - start) / segments;
        let data: Vec<T> = (0..n)
            .map(|i| base.pow(start + step * i.try_into().expect("n too large")))
            .collect();
        Ndarr::new(&data, [data.len()]).unwrap()
    }
    ///Generates a sequence of values that form a geometric progression.
    pub fn geomspace(start: T, end: T, n: u16) -> Self
    where
        u16: TryInto<T>,
        <u16 as TryInto<T>>::Error: Debug,
        T: Pow<T, Output = T> + Mul<Output = T> + Copy,
    {
        let segments: T = (n - 1).try_into().expect("n too large, max value 2^16");
        let step = (end - start) / segments;
        let exp: T = 1.try_into().expect("n too large") / segments;
        let ratio = (end / start).pow(exp);
        let data: Vec<T> = (0..n)
            .map(|i| start * ratio.pow(i.try_into().expect("n too large")))
            .collect();
        Ndarr::new(&data, [data.len()]).unwrap()
    }
}

#[cfg(test)]
mod mesh_test {
    use super::*;
    use crate::complex::C;
    #[test]
    fn linspace() {
        let x = Ndarr::linspace(0, 9, 10);
        assert_eq!(x, Ndarr::from(0..10))
    }

    #[test]
    fn logspace() {
        let x = Ndarr::logspace(0., 9., 10., 10);
        assert!(x.approx(&Ndarr::from([
            1., 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
        ])));
    }

    #[test]
    fn geomspace() {
        let x = Ndarr::geomspace(1., 256., 9);
        assert!(x.approx(&Ndarr::from([1., 2., 4., 8., 16., 32., 64., 128., 256.])));
    }
}
