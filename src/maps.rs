use super::*;

pub trait Bimap<F> {
    fn bimap(self, other: Self, f: F) -> Self;
    fn bimap_in_place(&mut self, other: Self, f: F);
}
//TODO: Here we need to think about if valueble maybe checking for the same shape and return an option instead
impl<F, T: Debug + Clone + Default, const R: usize> Bimap<F> for Ndarr<T, R>
where
    F: Fn(T, T) -> T,
{
    fn bimap(self, other: Self, f: F) -> Self {
        let mut out = vec![T::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(self.data[i].clone(), other.data[i].clone())
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }

    fn bimap_in_place(&mut self, other: Self, f: F) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i].clone(), other.data[i].clone())
        }
    }
}

trait GeneralBimap<F, T2, T3> {
    type Other;
    type Output;
    fn gen_bimap(self, other: Self::Other, f: F) -> Self::Output;
}

impl<F, T1, T2, T3, const R: usize> GeneralBimap<F, T2, T3> for Ndarr<T1, R>
where
    T1: Debug + Clone + Default,
    T2: Debug + Clone + Default,
    T3: Debug + Clone + Default,
    F: Fn(T1, T2) -> T3,
{
    type Other = Ndarr<T2, R>;
    type Output = Ndarr<T3, R>;

    fn gen_bimap(self, other: Self::Other, f: F) -> Self::Output {
        let mut out = vec![T3::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(self.data[i].clone(), other.data[i].clone())
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
}

pub trait Map<F> {
    fn map(self, f: F) -> Self;

    fn map_in_place(&mut self, f: F);
}

impl<F, T: Debug + Clone + Default, const R: usize> Map<F> for Ndarr<T, R>
where
    F: Fn(&T) -> T,
{
    fn map(self, f: F) -> Self {
        let mut out = vec![T::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(&self.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
    fn map_in_place(&mut self, f: F) {
        for val in self.data.iter_mut() {
            *val = f(val)
        }
    }
}