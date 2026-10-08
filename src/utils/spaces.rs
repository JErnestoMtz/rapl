use std::ops::{Add, Div, Sub};

use super::{Ndarr, ScanDirection, U1};
use std::fmt::Debug;

impl<T: Clone + Add<Output = T> + Div<Output = T> + Sub<Output = T>> Ndarr<T, U1> {
    /// Return evenly spaced numbers over a specified interval.
    pub fn linspace(start: T, end: T, n: u16) -> Self
    where
        u16: TryInto<T>,
        <u16 as TryInto<T>>::Error: Debug,
    {
        let segments: T = (n - 1).try_into().expect("n too large, max value 2^16");

        let dx = (end - start.clone()) / segments;
        // A running sum (APL `+\`): `start`, then `dx` repeated.
        Ndarr::from_fn([usize::from(n)], |ix| {
            if ix[0] == 0 {
                start.clone()
            } else {
                dx.clone()
            }
        })
        .scan_axis(0, ScanDirection::Forward, |sum, step| sum + step)
        .expect("a rank-one array has axis 0")
    }
}

#[cfg(test)]
mod mesh_test {
    use super::*;
    #[test]
    fn linspace() {
        let x = Ndarr::linspace(0, 9, 10);
        assert_eq!(x, Ndarr::from(0..10))
    }
}
