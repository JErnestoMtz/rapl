use super::*;
use num_traits::Signed;

impl<T: Signed, R: Rank, B: Buffer<T>> Ndarr<T, R, B> {
    pub fn abs(&self) -> Ndarr<T, R> {
        self.map(|x| x.abs())
    }
    pub fn is_positive(&self) -> Ndarr<bool, R> {
        self.map(|x| x.is_positive())
    }

    pub fn is_negative(&self) -> Ndarr<bool, R> {
        self.map(|x| x.is_negative())
    }
}
