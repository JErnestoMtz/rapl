use num_traits::Float;

use super::*;

impl<T, const R: usize> Ndarr<T, R>
where
    T: Clone + Copy + Debug + Default + Float,
{
    pub fn sin(&self) -> Self {
        self.map(|x| x.sin())
    }

    pub fn cos(&self) -> Self {
        self.map(|x| x.cos())
    }

    pub fn tan(&self) -> Self {
        self.map(|x| x.tan())
    }

    pub fn sinh(&self) -> Self {
        self.map(|x| x.sinh())
    }

    pub fn cosh(&self) -> Self {
        self.map(|x| x.cosh())
    }

    pub fn tanh(&self) -> Self {
        self.map(|x| x.tanh())
    }

    pub fn exp(&self) -> Self {
        self.map(|x| x.exp())
    }

    pub fn log(&self, base: T) -> Self {
        self.map(|x| x.log(base))
    }

    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    pub fn log2(&self) -> Self {
        self.map(|x| x.log2())
    }
    pub fn is_infinite(&self) -> Ndarr<bool, R> {
        self.map_types(|x| x.is_infinite())
    }
    pub fn is_finite(&self) -> Ndarr<bool, R> {
        self.map_types(|x| x.is_finite())
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
        self.map_types(|x| x.is_normal())
    }
    pub fn is_nan(&self) -> Ndarr<bool, R> {
        self.map_types(|x| x.is_nan())
    }

    ///Max Float, floating types do not implement `Ord`, but this gives a way to get the maximum value in an `Ndarr` if all comparisons are allowed.
    pub fn maxf(&self)->T{
        self.data.clone().into_iter().reduce(T::max).expect("Cannot perform fmax deu to imposable comparison")
    }
    ///Min Float, floating types do not implement `Ord`, but this gives a way to get the minimum value in an `Ndarr` if all comparisons are allowed.
    pub fn minf(&self)->T{
        self.data.clone().into_iter().reduce(T::min).expect("Cannot perform minf due to imposable comparison.")
    }
}
