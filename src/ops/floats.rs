use num_traits::Float;

use super::*;

// Shares method names with the complex family (`complex_tensor.rs`); the element
// type as head parameter lets rustc prove the inherent impls disjoint.
impl<T: Float, R: Rank, B: Buffer<T>> Ndarr<T, R, B> {
    pub fn sin(&self) -> Ndarr<T, R> {
        self.map(|x| x.sin())
    }

    pub fn cos(&self) -> Ndarr<T, R> {
        self.map(|x| x.cos())
    }

    pub fn tan(&self) -> Ndarr<T, R> {
        self.map(|x| x.tan())
    }

    pub fn sinh(&self) -> Ndarr<T, R> {
        self.map(|x| x.sinh())
    }

    pub fn cosh(&self) -> Ndarr<T, R> {
        self.map(|x| x.cosh())
    }

    pub fn tanh(&self) -> Ndarr<T, R> {
        self.map(|x| x.tanh())
    }

    pub fn exp(&self) -> Ndarr<T, R> {
        self.map(|x| x.exp())
    }

    pub fn log(&self, base: T) -> Ndarr<T, R> {
        self.map(|x| x.log(base))
    }

    pub fn ln(&self) -> Ndarr<T, R> {
        self.map(|x| x.ln())
    }

    pub fn log2(&self) -> Ndarr<T, R> {
        self.map(|x| x.log2())
    }
    pub fn is_infinite(&self) -> Ndarr<bool, R> {
        self.map(|x| x.is_infinite())
    }
    pub fn is_finite(&self) -> Ndarr<bool, R> {
        self.map(|x| x.is_finite())
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
        self.map(|x| x.is_normal())
    }
    pub fn is_nan(&self) -> Ndarr<bool, R> {
        self.map(|x| x.is_nan())
    }

    /// Whether two arrays have the same shape and all elements differ by at most `tolerance`.
    pub fn approx_epsilon<R2: Rank, B2: Buffer<T>>(
        &self,
        other: &Ndarr<T, R2, B2>,
        tolerance: T,
    ) -> bool {
        self.shape() == other.shape()
            && self
                .iter_elems()
                .zip(other.iter_elems())
                .all(|(left, right)| (*left - *right).abs() <= tolerance)
    }
}
